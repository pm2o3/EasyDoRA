"""
EasyDora Configuration
Hardcoded optimal parameters for IllustriousXL character LoRA training.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration - optimized for character LoRAs."""

    # === Paths ===
    base_model_dir: Path = Path("BaseModel")
    dataset_dir: Path = Path("Dataset")
    output_dir: Path = Path("Outputs")
    cache_dir: Path = Path("Cache")

    # === Character LoRA Settings ===
    trigger_word: str = ""  # Set at runtime via CLI arg, file, or prompt

    shuffle_tags: bool = True
    keep_tokens: int = 1  # Preserve first N comma-separated tokens (trigger word) during shuffle

    caption_dropout_rate: float = 0.0   # Disabled - character LoRAs should only activate via trigger word

    # === Adapter Parameters ===
    # Adapter type - "loha" (Hadamard product), "lora" (standard), or "dora" (decomposed magnitude/direction)
    adapter_type: str = "dora"

    lora_rank: int = 32

    lora_alpha: int = 16

    lora_target_modules: list[str] = None  # Will be set in __post_init__

    # Train text encoders with adapter (critical for trigger word association)
    lora_text_encoder: bool = True
    lora_text_encoder_rank: int = 16
    lora_text_encoder_alpha: int = 8

    # === Loss Weighting ===
    min_snr_gamma: float = 5.0

    # === Training Parameters ===
    # Optimizer - "prodigy" (auto-LR, recommended) or "adamw8bit" (manual LR)
    optimizer: str = "prodigy"

    # Learning rates - only used if optimizer="adamw8bit"
    unet_learning_rate: float = 2e-4
    text_encoder_learning_rate: float = 2e-5

    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 2

    # Training duration
    num_epochs: int = 22
    checkpoint_every_n_epochs: int = 1  # Save every epoch (character LoRAs overfit fast)

    mixed_precision: str = "bf16"

    # Learning rate schedule
    lr_scheduler: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = True
    enable_xformers: bool = True

    # VRAM optimization
    offload_vae: bool = True

    # Dataset
    resolution: int = 1024  # Base resolution, aspect ratios handled automatically
    num_repeats: int = 2  # Repeat each image N times per epoch 

    # Seed for reproducibility
    seed: int = 42

    # === Sample Generation ===
    sample_every_n_epochs: int = 1  # 0 to disable
    sample_prompt: str = ""
    sample_negative_prompt: str = ""
    sample_steps: int = 25
    sample_cfg_scale: float = 6.0

    # Reforge/Forge API settings - launch Reforge with --api flag
    # and add Outputs/ folder to its LoRA search paths (--lora-dir or symlink)
    sample_api_url: str = "http://127.0.0.1:7860"  # Empty string = skip API, use onboard only
    sample_lora_strength: float = 1.0

    # === Prodigy Optimizer Tuning ===
    prodigy_d_coef: float = 0.4
    prodigy_beta2: float = 0.99
    prodigy_safeguard_warmup: bool = True
    prodigy_use_bias_correction: bool = True

    def __post_init__(self):
        if self.lora_target_modules is None:
            object.__setattr__(self, 'lora_target_modules', [
                "to_q", "to_k",
                "to_v",
            ])


CONFIG = TrainingConfig()
