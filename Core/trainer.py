"""
EasyDora Training Loop
LoRA training for IllustriousXL with character focus.
Text encoding happens at runtime through LoRA-wrapped encoders for proper
trigger word learning and tag shuffling support.
"""
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft import LoHaConfig

from .config import CONFIG
from .dataset import ImageCaptionDataset, AspectRatioBucketSampler, collate_fn


class IllustriousLoraTrainer:
    """
    Minimal trainer for IllustriousXL character LoRA training.
    Optimized for character training on RTX 5070 TI.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Paths
        self.base_model_path = Path(CONFIG.base_model_dir)
        self.output_path = Path(CONFIG.output_dir)
        self.output_path.mkdir(exist_ok=True)

        # Will be initialized in setup()
        self.unet = None
        self.vae = None
        self.text_encoder_one = None
        self.text_encoder_two = None
        self.tokenizer_one = None
        self.tokenizer_two = None
        self.noise_scheduler = None
        self.optimizers = []       # List of optimizers (split UNet/TE for Prodigy)
        self.lr_schedulers = []    # Matching list of LR schedulers
        self.dataloader = None

    def setup(self):
        """Initialize all model components."""
        print("=" * 60)
        print("EasyDora - IllustriousXL Character LoRA Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Precision: {CONFIG.mixed_precision}")
        print(f"Base model: {self.base_model_path}")
        print()
        print(f"Trigger word: '{CONFIG.trigger_word}'")
        print(f"Tag shuffling: {CONFIG.shuffle_tags} (keep first {CONFIG.keep_tokens} tokens)")
        print(f"Caption dropout: {CONFIG.caption_dropout_rate:.0%}")
        print(f"LoRA rank: {CONFIG.lora_rank}, alpha: {CONFIG.lora_alpha} (scale: {CONFIG.lora_alpha / CONFIG.lora_rank:.2f})")
        print(f"TE LoRA rank: {CONFIG.lora_text_encoder_rank}, alpha: {CONFIG.lora_text_encoder_alpha} (scale: {CONFIG.lora_text_encoder_alpha / CONFIG.lora_text_encoder_rank:.2f})")
        if CONFIG.min_snr_gamma > 0:
            print(f"min-SNR loss weighting: gamma={CONFIG.min_snr_gamma}")
        else:
            print("min-SNR loss weighting: disabled")
        print()

        self._load_models()
        self._inject_lora()
        self._setup_training()
        self._setup_dataset()
        self._setup_optimizer()

        print("=" * 60)
        print("Setup complete!")
        print(f"Dataset size: {len(self.dataloader.dataset)} images")
        print(f"Batch size: {CONFIG.batch_size} x {CONFIG.gradient_accumulation_steps} accumulation")
        print(f"Epochs: {CONFIG.num_epochs}")
        print(f"Checkpoints: Every {CONFIG.checkpoint_every_n_epochs} epoch(s)")
        if CONFIG.sample_every_n_epochs > 0:
            mode = f"API → {CONFIG.sample_api_url}" if CONFIG.sample_api_url else "onboard"
            print(f"Sample generation: Every {CONFIG.sample_every_n_epochs} epoch(s) ({mode})")
        else:
            print("Sample generation: disabled")
        print("=" * 60)
        print()

    def _load_models(self):
        """Load all model components from base model (safetensors or diffusers format)."""
        print("Loading models...")

        # Check for single safetensors file first
        safetensor_files = list(self.base_model_path.glob("*.safetensors"))

        if safetensor_files:
            # Load directly from single safetensors file
            safetensor_path = safetensor_files[0]
            print(f"  Loading from: {safetensor_path.name}")

            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(safetensor_path),
                torch_dtype=self.dtype,
            )

            # Extract components from pipeline
            self.vae = pipe.vae.to(self.device)
            self.unet = pipe.unet.to(self.device)
            self.text_encoder_one = pipe.text_encoder.to(self.device)
            self.text_encoder_two = pipe.text_encoder_2.to(self.device)
            self.tokenizer_one = pipe.tokenizer
            self.tokenizer_two = pipe.tokenizer_2

            # IMPORTANT: Use DDPMScheduler for training, not the inference scheduler
            self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

            # Free the pipeline wrapper
            del pipe
            torch.cuda.empty_cache()
        else:
            # Fall back to diffusers format (subfolders)
            model_path = str(self.base_model_path)

            print("  Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", torch_dtype=self.dtype
            ).to(self.device)

            print("  Loading text encoders...")
            self.tokenizer_one = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer"
            )
            self.tokenizer_two = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer_2"
            )
            self.text_encoder_one = CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=self.dtype
            ).to(self.device)
            self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                model_path, subfolder="text_encoder_2", torch_dtype=self.dtype
            ).to(self.device)

            print("  Loading UNet...")
            self.unet = UNet2DConditionModel.from_pretrained(
                model_path, subfolder="unet", torch_dtype=self.dtype
            ).to(self.device)

            self.noise_scheduler = DDPMScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )

        # VAE is never trained
        self.vae.requires_grad_(False)
        self.vae.eval()

        print("  Models loaded!")

    def _make_adapter_config(self, rank: int, alpha: int, target_modules: list[str]):
        """Create adapter config based on CONFIG.adapter_type."""
        if CONFIG.adapter_type == "loha":
            return LoHaConfig(
                r=rank,
                alpha=alpha,
                target_modules=target_modules,
                rank_dropout=0.0,  # Must be 0 with Prodigy optimizer
                module_dropout=0.0,
            )
        elif CONFIG.adapter_type == "dora":
            return LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.0,  # Must be 0 with Prodigy optimizer
                bias="none",
                use_dora=True,
            )
        else:
            return LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.0,  # Must be 0 with Prodigy optimizer
                bias="none",
            )

    def _inject_lora(self):
        """Inject adapter layers into UNet and text encoders."""
        adapter_name = CONFIG.adapter_type.upper()
        print(f"Injecting {adapter_name} layers...")

        # Configure adapter for UNet
        unet_config = self._make_adapter_config(
            CONFIG.lora_rank, CONFIG.lora_alpha, CONFIG.lora_target_modules,
        )
        self.unet = get_peft_model(self.unet, unet_config)
        trainable_unet = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"  UNet {adapter_name} injected: {trainable_unet:,} trainable parameters")

        # Configure adapter for text encoders (critical for trigger word learning)
        if CONFIG.lora_text_encoder:
            te_config = self._make_adapter_config(
                CONFIG.lora_text_encoder_rank,
                CONFIG.lora_text_encoder_alpha,
                ["q_proj", "k_proj", "v_proj", "out_proj"],  # CLIP attention layers
            )

            self.text_encoder_one = get_peft_model(self.text_encoder_one, te_config)
            self.text_encoder_two = get_peft_model(self.text_encoder_two, te_config)

            trainable_te1 = sum(p.numel() for p in self.text_encoder_one.parameters() if p.requires_grad)
            trainable_te2 = sum(p.numel() for p in self.text_encoder_two.parameters() if p.requires_grad)
            print(f"  Text Encoder 1 {adapter_name} injected: {trainable_te1:,} trainable parameters")
            print(f"  Text Encoder 2 {adapter_name} injected: {trainable_te2:,} trainable parameters")
        else:
            # Freeze text encoders completely
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_two.requires_grad_(False)
            print("  Text encoders frozen (no adapter)")

    def _setup_training(self):
        """Configure models for training."""
        print("Configuring for training...")

        # Enable gradient checkpointing for memory efficiency
        if CONFIG.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if CONFIG.lora_text_encoder:
                # PEFT models use a different method
                self.text_encoder_one.base_model.model.gradient_checkpointing_enable()
                self.text_encoder_two.base_model.model.gradient_checkpointing_enable()

        # Enable memory efficient attention
        if CONFIG.enable_xformers:
            try:
                # Access the base UNet model when using PEFT
                self.unet.base_model.model.enable_xformers_memory_efficient_attention()
                print("  xformers enabled")
            except Exception:
                # Fall back to PyTorch SDPA (built into PyTorch 2.0+)
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.unet.base_model.model.set_attn_processor(AttnProcessor2_0())
                print("  Using PyTorch SDPA attention (memory efficient)")

        # Set training mode
        self.unet.train()
        if CONFIG.lora_text_encoder:
            self.text_encoder_one.train()
            self.text_encoder_two.train()
        else:
            self.text_encoder_one.eval()
            self.text_encoder_two.eval()

    def _setup_dataset(self):
        """Initialize dataset and dataloader."""
        print("Setting up dataset...")

        # Dataset only needs VAE for latent caching
        dataset = ImageCaptionDataset(
            vae=self.vae,
            device=self.device,
        )

        # Offload VAE to CPU after caching (saves ~1.5GB VRAM)
        if CONFIG.offload_vae:
            self.vae = self.vae.to("cpu")
            torch.cuda.empty_cache()
            print("  VAE offloaded to CPU")

        # Use bucket sampler to group same-sized images together
        bucket_sampler = AspectRatioBucketSampler(
            dataset,
            batch_size=CONFIG.batch_size,
            shuffle=True,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_sampler=bucket_sampler,  # Yields batches of indices
            collate_fn=collate_fn,
            num_workers=0,  # Keep simple, data is cached
            pin_memory=True,
        )

    def _setup_optimizer(self):
        """Initialize optimizer - Prodigy (auto-LR) or AdamW8bit (manual LR)."""
        print("Setting up optimizer...")

        # Collect trainable parameters by component
        unet_params = [p for p in self.unet.parameters() if p.requires_grad]

        te_params = []
        if CONFIG.lora_text_encoder:
            te_params = (
                [p for p in self.text_encoder_one.parameters() if p.requires_grad]
                + [p for p in self.text_encoder_two.parameters() if p.requires_grad]
            )

        num_training_steps = (
            len(self.dataloader) * CONFIG.num_epochs
            // CONFIG.gradient_accumulation_steps
        )
        warmup_steps = min(100, max(10, num_training_steps // 20))

        # Select optimizer based on config
        if CONFIG.optimizer == "prodigy":
            from prodigyopt import Prodigy

            opt_unet = Prodigy(
                unet_params,
                lr=1.0,
                betas=(0.9, CONFIG.prodigy_beta2),
                beta3=None,
                weight_decay=0.01,
                eps=1e-8,
                decouple=True,
                d_coef=CONFIG.prodigy_d_coef,
                safeguard_warmup=CONFIG.prodigy_safeguard_warmup,
                use_bias_correction=CONFIG.prodigy_use_bias_correction,
            )
            self.optimizers.append(opt_unet)
            print(f"  UNet Prodigy: d_coef={CONFIG.prodigy_d_coef}, beta2={CONFIG.prodigy_beta2}")

            if te_params:
                opt_te = Prodigy(
                    te_params,
                    lr=1.0,
                    betas=(0.9, CONFIG.prodigy_beta2),
                    beta3=None,
                    weight_decay=0.01,
                    eps=1e-8,
                    decouple=True,
                    d_coef=CONFIG.prodigy_d_coef,
                    safeguard_warmup=CONFIG.prodigy_safeguard_warmup,
                    use_bias_correction=CONFIG.prodigy_use_bias_correction,
                )
                self.optimizers.append(opt_te)
                print(f"  TE Prodigy: d_coef={CONFIG.prodigy_d_coef}")

            # Cosine schedule on top of Prodigy - decays the LR multiplier over training
            for opt in self.optimizers:
                self.lr_schedulers.append(get_scheduler(
                    "cosine",
                    optimizer=opt,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                ))
            print(f"  LR schedule: cosine with {warmup_steps} warmup steps ({num_training_steps} total)")

        elif CONFIG.optimizer == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit

            # AdamW8bit supports per-group LR natively, so one instance is fine
            param_groups = [{"params": unet_params, "lr": CONFIG.unet_learning_rate}]
            if te_params:
                param_groups.append({"params": te_params, "lr": CONFIG.text_encoder_learning_rate})

            opt = AdamW8bit(param_groups, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
            self.optimizers.append(opt)
            print("  Using AdamW8bit optimizer")

            self.lr_schedulers.append(get_scheduler(
                CONFIG.lr_scheduler,
                optimizer=opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            ))
            print(f"  LR schedule: {CONFIG.lr_scheduler} with {warmup_steps} warmup steps ({num_training_steps} total)")

        else:
            raise ValueError(f"Unknown optimizer: {CONFIG.optimizer}. Use 'prodigy' or 'adamw8bit'.")

    def _encode_batch_prompts(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of captions using both SDXL text encoders (with LoRA).

        Unlike the old approach that cached embeddings using the base model,
        this encodes through the LoRA-wrapped text encoders so they receive
        direct gradients and properly learn the trigger word association.
        """
        # Tokenize entire batch at once (tokenizers natively handle lists)
        tokens_one = self.tokenizer_one(
            captions,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        tokens_two = self.tokenizer_two(
            captions,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Single forward pass per encoder for entire batch
        encoder_output_one = self.text_encoder_one(tokens_one, output_hidden_states=True)
        encoder_output_two = self.text_encoder_two(tokens_two, output_hidden_states=True)

        # SDXL uses penultimate hidden states, concatenated from both encoders
        prompt_embeds = torch.cat([
            encoder_output_one.hidden_states[-2],
            encoder_output_two.hidden_states[-2],
        ], dim=-1)

        # Pooled output from second encoder
        pooled_prompt_embeds = encoder_output_two[0]

        return prompt_embeds, pooled_prompt_embeds

    def _get_time_ids(self, batch: dict) -> torch.Tensor:
        """Compute SDXL time IDs for batch."""
        add_time_ids = []
        for i in range(len(batch["original_sizes"])):
            original_size = batch["original_sizes"][i]
            crop_coords = batch["crop_coords"][i]
            target_size = batch["target_sizes"][i]

            add_time_ids.append(list(original_size) + list(crop_coords) + list(target_size))

        return torch.tensor(add_time_ids, dtype=self.dtype, device=self.device)

    def _compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio for given timesteps.

        SNR = alpha^2 / sigma^2 from the noise scheduler's cumulative schedule.
        Used by min-SNR loss weighting to downweight easy (low-noise) timesteps.
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=torch.float32
        )
        alpha = alphas_cumprod[timesteps] ** 0.5
        sigma = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        return (alpha / sigma) ** 2

    def _training_step(self, batch: dict) -> torch.Tensor:
        """Execute single training step."""
        # Move latents to device
        latents = batch["latents"].to(device=self.device, dtype=self.dtype)

        # Encode captions at runtime through LoRA text encoders
        prompt_embeds, pooled_prompt_embeds = self._encode_batch_prompts(batch["captions"])
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.dtype)

        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device, dtype=torch.long
        )

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get time IDs for SDXL
        add_time_ids = self._get_time_ids(batch)

        # Prepare added conditions
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # Predict noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # Compute per-sample MSE loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
        loss = loss.mean(dim=[1, 2, 3])  # [B, C, H, W] -> [B] per-sample

        # min-SNR weighting: downweight easy (low-noise) timesteps
        if CONFIG.min_snr_gamma > 0:
            snr = self._compute_snr(timesteps)
            loss = loss * torch.clamp(snr, max=CONFIG.min_snr_gamma) / snr

        loss = loss.mean()

        return loss

    # PEFT adapter key names → Kohya format names
    _PEFT_TO_KOHYA = {
        # Standard LoRA / DoRA
        "lora_A": "lora_down",
        "lora_B": "lora_up",
        "lora_magnitude_vector": "dora_scale",  # DoRA magnitude component
        # LoHa (Hadamard product) - Kohya uses same names
        "hada_w1_a": "hada_w1_a",
        "hada_w1_b": "hada_w1_b",
        "hada_w2_a": "hada_w2_a",
        "hada_w2_b": "hada_w2_b",
    }
    # Save alpha once per module on the first key encountered
    _ALPHA_TRIGGERS = {"lora_A", "hada_w1_a"}

    def _collect_adapter_weights(self, model, prefix: str, alpha: float, state_dict: dict):
        """Extract adapter weights from a PEFT model into Kohya-compatible state dict.

        Handles multiple PEFT key formats:
          LoRA:  <path>.lora_A.weight          (nn.Linear → .weight suffix)
          LoHa:  <path>.hada_w1_a.default      (ParameterDict → adapter name suffix)
          LoHa:  <path>.hada_w1_a              (bare nn.Parameter → no suffix)
        """
        peft_sd = get_peft_model_state_dict(model)
        seen_modules = set()

        for key, value in peft_sd.items():
            key = key.replace("base_model.model.", "")
            parts = key.split(".")

            # Search backwards for a known adapter key name.
            adapter_key = None
            adapter_idx = None
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] in self._PEFT_TO_KOHYA:
                    adapter_key = parts[i]
                    adapter_idx = i
                    break

            if adapter_key is None:
                continue

            module_path = "_".join(parts[:adapter_idx])
            kohya_name = self._PEFT_TO_KOHYA[adapter_key]

            # Kohya format: LoRA uses .weight suffix (nn.Linear), LoHa/DoRA magnitude doesn't (nn.Parameter)
            if adapter_key.startswith("lora_") and adapter_key != "lora_magnitude_vector":
                kohya_key = f"{prefix}_{module_path}.{kohya_name}.weight"
            else:
                kohya_key = f"{prefix}_{module_path}.{kohya_name}"

            weight = value.contiguous()

            # DoRA magnitude vector: PEFT stores as 1D [out_features] but Forge/ComfyUI's
            # weight_decompose needs 2D [out_features, 1] for correct broadcasting against
            # weight_norm [out_features, 1]. Without this, 1D broadcasts as [1, out_features]
            # and produces a wrong [out_features, out_features] cross-product, which crashes
            # on non-square weights (cross-attention to_k) and silently gives wrong results
            # on square weights (self-attention).
            if adapter_key == "lora_magnitude_vector" and weight.dim() == 1:
                weight = weight.unsqueeze(-1)

            state_dict[kohya_key] = weight.to(dtype=torch.float16, device="cpu")

            # Save alpha once per module
            if adapter_key in self._ALPHA_TRIGGERS and module_path not in seen_modules:
                seen_modules.add(module_path)
                state_dict[f"{prefix}_{module_path}.alpha"] = torch.tensor(float(alpha))

    def save_checkpoint(self, epoch: int):
        """Save adapter checkpoint in Kohya/ComfyUI compatible format."""
        output_file = self.output_path / f"lora_epoch_{epoch:02d}.safetensors"

        print(f"\nSaving checkpoint: {output_file}")

        state_dict = {}

        # UNet adapter weights
        self._collect_adapter_weights(self.unet, "lora_unet", CONFIG.lora_alpha, state_dict)

        # Text encoder adapter weights (if enabled)
        if CONFIG.lora_text_encoder:
            self._collect_adapter_weights(
                self.text_encoder_one, "lora_te1", CONFIG.lora_text_encoder_alpha, state_dict,
            )
            self._collect_adapter_weights(
                self.text_encoder_two, "lora_te2", CONFIG.lora_text_encoder_alpha, state_dict,
            )

        print(f"  Collected {len(state_dict)} adapter tensors")

        # Save as safetensors
        save_file(state_dict, output_file)

        # Report file size
        size_mb = output_file.stat().st_size / (1024**2)
        print(f"Checkpoint saved! ({size_mb:.2f} MB)")

    def _build_sample_prompt(self) -> tuple[str, str]:
        """Build prompt with trigger word prepended."""
        prompt = CONFIG.sample_prompt
        if not prompt:
            prompt = "1girl, solo, upper body, looking at viewer, simple background, white background"

        # Prepend trigger word if not already present
        if CONFIG.trigger_word and CONFIG.trigger_word not in prompt:
            prompt = f"{CONFIG.trigger_word}, {prompt}"

        return prompt, CONFIG.sample_negative_prompt

    def _generate_sample(self, epoch: int):
        """Generate a test sample — tries Reforge API first, falls back to onboard."""
        sample_dir = self.output_path / "samples"
        sample_dir.mkdir(exist_ok=True)

        if CONFIG.sample_api_url:
            if self._generate_sample_api(epoch, sample_dir):
                return
            print("  Falling back to onboard inference...")

        self._generate_sample_onboard(epoch, sample_dir)

    @staticmethod
    def _make_sample_grid(images, cols: int = 2):
        """Assemble PIL images into a grid. Returns a single PIL image."""
        from PIL import Image as PILImage
        rows = (len(images) + cols - 1) // cols
        w, h = images[0].size
        grid = PILImage.new("RGB", (w * cols, h * rows))
        for i, img in enumerate(images):
            grid.paste(img, (w * (i % cols), h * (i // cols)))
        return grid

    def _generate_sample_api(self, epoch: int, sample_dir: Path) -> bool:
        """Generate 4 samples via Reforge/Forge API as a 2x2 grid. Returns True on success."""
        import requests
        import base64
        import io
        from PIL import Image as PILImage

        api_url = CONFIG.sample_api_url.rstrip("/")
        prompt, neg_prompt = self._build_sample_prompt()
        lora_name = f"lora_epoch_{epoch:02d}"

        # Wrap prompt with LoRA loading syntax for Reforge
        full_prompt = f"<lora:{lora_name}:{CONFIG.sample_lora_strength}> {prompt}"

        print(f"  Generating 4 samples via API: \"{full_prompt}\"")

        try:
            # Refresh LoRA list so Reforge sees the new checkpoint
            requests.post(f"{api_url}/sdapi/v1/refresh-loras", timeout=10)

            payload = {
                "prompt": full_prompt,
                "negative_prompt": neg_prompt,
                "steps": CONFIG.sample_steps,
                "cfg_scale": CONFIG.sample_cfg_scale,
                "width": 1024,
                "height": 1024,
                "seed": CONFIG.seed,
                "batch_size": 4,
                "sampler_name": "Euler",
            }

            response = requests.post(
                f"{api_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()

            result = response.json()
            images = []
            for img_b64 in result["images"][:4]:
                img_data = base64.b64decode(img_b64)
                images.append(PILImage.open(io.BytesIO(img_data)))

            grid = self._make_sample_grid(images)
            sample_path = sample_dir / f"sample_e{epoch:02d}.png"
            grid.save(sample_path)

            print(f"  Sample grid saved: {sample_path}")
            return True

        except requests.exceptions.ConnectionError:
            print(f"  Warning: Could not connect to {api_url}. Is Reforge running with --api?")
            return False
        except Exception as e:
            print(f"  Warning: API sample generation failed: {e}")
            return False

    def _generate_sample_onboard(self, epoch: int, sample_dir: Path):
        """Generate 4 samples using the training model as a 2x2 grid (fallback).

        Each sample uses a different seed (seed, seed+1, ...) but seeds are fixed
        across epochs so grids are directly comparable.
        """
        from PIL import Image as PILImage
        import numpy as np

        num_samples = 4
        prompt, neg_prompt = self._build_sample_prompt()
        print(f"  Generating {num_samples} samples (onboard): \"{prompt}\"")

        # Switch to eval mode
        self.unet.eval()
        if CONFIG.lora_text_encoder:
            self.text_encoder_one.eval()
            self.text_encoder_two.eval()

        # Bring VAE back to GPU for decoding
        vae_was_offloaded = next(self.vae.parameters()).device.type == "cpu"
        if vae_was_offloaded:
            self.vae.to(self.device)

        try:
            with torch.no_grad():
                # Inference scheduler
                scheduler = EulerDiscreteScheduler.from_config(self.noise_scheduler.config)
                scheduler.set_timesteps(CONFIG.sample_steps, device=self.device)

                # Encode prompt and negative through LoRA text encoders (single encode, expand to batch)
                prompt_embeds, pooled_prompt_embeds = self._encode_batch_prompts([prompt])
                prompt_embeds = prompt_embeds.to(dtype=self.dtype).expand(num_samples, -1, -1)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.dtype).expand(num_samples, -1)

                neg_embeds, neg_pooled = self._encode_batch_prompts([neg_prompt])
                neg_embeds = neg_embeds.to(dtype=self.dtype).expand(num_samples, -1, -1)
                neg_pooled = neg_pooled.to(dtype=self.dtype).expand(num_samples, -1)

                # SDXL time IDs expanded to batch
                add_time_ids = torch.tensor(
                    [[1024, 1024, 0, 0, 1024, 1024]], dtype=self.dtype, device=self.device
                ).expand(num_samples, -1)

                # Different seed per sample for variety, fixed across epochs for comparison
                latents_list = []
                for i in range(num_samples):
                    gen = torch.Generator(device=self.device).manual_seed(CONFIG.seed + i)
                    latents_list.append(torch.randn(
                        (1, 4, 128, 128), generator=gen,
                        device=self.device, dtype=self.dtype,
                    ))
                latents = torch.cat(latents_list)  # [4, 4, 128, 128]
                latents = latents * scheduler.init_noise_sigma

                # Denoising loop — batch 4 images with CFG (effective UNet batch = 8)
                for t in scheduler.timesteps:
                    latent_input = torch.cat([latents] * 2)  # [8, 4, 128, 128]
                    latent_input = scheduler.scale_model_input(latent_input, t)

                    noise_pred = self.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=torch.cat([neg_embeds, prompt_embeds]),
                        added_cond_kwargs={
                            "text_embeds": torch.cat([neg_pooled, pooled_prompt_embeds]),
                            "time_ids": torch.cat([add_time_ids] * 2),
                        },
                        return_dict=False,
                    )[0]

                    # Apply CFG
                    pred_uncond, pred_text = noise_pred.chunk(2)
                    noise_pred = pred_uncond + CONFIG.sample_cfg_scale * (pred_text - pred_uncond)

                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Decode all 4 latents → pixels (VAE needs float32)
                self.vae.to(dtype=torch.float32)
                decoded = self.vae.decode(
                    latents.to(dtype=torch.float32) / self.vae.config.scaling_factor,
                    return_dict=False,
                )[0]
                self.vae.to(dtype=self.dtype)

                # Convert batch to PIL images and assemble grid
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                pixels = decoded.cpu().permute(0, 2, 3, 1).float().numpy()  # [4, H, W, 3]
                pixels = (pixels * 255).round().astype(np.uint8)

                images = [PILImage.fromarray(pixels[i]) for i in range(num_samples)]
                grid = self._make_sample_grid(images)

                sample_path = sample_dir / f"sample_e{epoch:02d}.png"
                grid.save(sample_path)
                print(f"  Sample grid saved: {sample_path}")

        finally:
            # Restore training state
            self.unet.train()
            if CONFIG.lora_text_encoder:
                self.text_encoder_one.train()
                self.text_encoder_two.train()

            # Re-offload VAE
            if vae_was_offloaded:
                self.vae.to("cpu")
                torch.cuda.empty_cache()

    def _clip_and_step(self):
        """Clip gradients and step all optimizers/schedulers."""
        # Clip UNet and TE gradients separately (each optimizer owns its params)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.unet.parameters() if p.requires_grad], 1.0
        )
        if CONFIG.lora_text_encoder:
            # Clip TE1+TE2 together (they share one optimizer when split)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.text_encoder_one.parameters() if p.requires_grad]
                + [p for p in self.text_encoder_two.parameters() if p.requires_grad],
                1.0
            )

        for opt in self.optimizers:
            opt.step()
        for sched in self.lr_schedulers:
            sched.step()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def train(self):
        """Main training loop."""
        print("Starting training...")
        print()

        global_step = 0

        for epoch in range(1, CONFIG.num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            # Progress bar for epoch
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{CONFIG.num_epochs}",
                leave=True,
            )

            for step, batch in enumerate(pbar):
                # Accumulate gradients
                loss = self._training_step(batch)
                loss = loss / CONFIG.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * CONFIG.gradient_accumulation_steps
                num_batches += 1

                # Update weights every gradient_accumulation_steps
                if (step + 1) % CONFIG.gradient_accumulation_steps == 0:
                    self._clip_and_step()
                    global_step += 1

                # Update progress bar
                avg_loss = epoch_loss / num_batches
                current_lr = self.lr_schedulers[0].get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            # Flush remaining accumulated gradients at epoch end
            if num_batches % CONFIG.gradient_accumulation_steps != 0:
                self._clip_and_step()
                global_step += 1

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch} complete - Average loss: {avg_epoch_loss:.4f}")

            # Save checkpoint every N epochs
            if epoch % CONFIG.checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch)

            # Generate sample image for visual progress tracking
            if CONFIG.sample_every_n_epochs > 0 and epoch % CONFIG.sample_every_n_epochs == 0:
                self._generate_sample(epoch)

            # Clear cache between epochs
            gc.collect()
            torch.cuda.empty_cache()

        # Save final checkpoint (only if not already saved)
        print("\nTraining complete!")
        if CONFIG.num_epochs % CONFIG.checkpoint_every_n_epochs != 0:
            self.save_checkpoint(CONFIG.num_epochs)
        print(f"\nAll LoRA checkpoints saved to: {self.output_path}")


def run_training():
    """Entry point for training."""
    # Set seed for reproducibility
    torch.manual_seed(CONFIG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.seed)

    trainer = IllustriousLoraTrainer()
    trainer.setup()
    trainer.train()
