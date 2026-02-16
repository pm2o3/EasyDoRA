# EasyDora

The only open-source DoRA character LoRA trainer that actually works at inference. Minimal, GUI-less, one command — specify a trigger word and train.

Built for **IllustriousXL** variants (anime SDXL models). Optimized for **RTX 5070 TI** (16GB VRAM). Produces production-ready `.safetensors` checkpoints compatible with ComfyUI, Forge, Reforge, and A1111.

## Why EasyDora?

Every other open-source DoRA implementation (including Kohya's) ships `dora_scale` as a **1D tensor**. Forge/Reforge's `weight_decompose` divides by a `[out, 1]` weight norm — a 1D `[N]` vector broadcasts as `[1, N]`, creating an `[N, N]` cross-product that silently corrupts square weight matrices and crashes on non-square ones (like cross-attention `to_k`). We fix this by saving `dora_scale` as 2D `[N, 1]`.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place base model .safetensors in BaseModel/
# Use an IllustriousXL variant that matches your dataset's style

# 3. Add your dataset to Dataset/
# Images (.png, .jpg, .webp) + matching .txt captions (Danbooru tags)

# 4. Train
python train.py mycharname

# 5. (Optional) If you wish to test samples on a model of your choice, with --api enabled in forge/reforge/forgeNeo/A1111 & with your localhost added to the config file, also add --lora-dir to arg line. For example "--api --lora-dir "C:\Users\X\Desktop\EasyDora\Outputs" if you're a desktop hog.
```

That's it. Best checkpoint is typically epoch 15-19. Built for 16gb vram. 

### Trigger Word Options

```bash
python train.py chr_sakura          # CLI argument (recommended)
echo "chr_sakura" > Dataset/trigger_word.txt && python train.py  # File
python train.py                     # Interactive prompt
```

Pick something **rare and unique** — `chr_mychar`, `sks_name`, `r0gue_oc`. Avoid real words or existing Danbooru tags.

## Folder Structure

```
EasyDora/
├── BaseModel/       <- Drop your .safetensors here
├── Dataset/         <- Images (.png) + captions (.txt)
│   └── trigger_word.txt  <- (optional) trigger word file
├── Cache/           <- VAE latent cache (auto-generated)
├── Outputs/         <- LoRA checkpoints + sample grids
│   └── samples/     <- Per-epoch 2x2 sample grids
├── Core/
│   ├── config.py    <- All training parameters
│   ├── dataset.py   <- Dataset loader with tag shuffle
│   └── trainer.py   <- Training loop, adapters, checkpointing
└── train.py         <- Entry point
```

## Research Findings

Five days of systematic A/B testing produced the following insights. Most of this isn't documented anywhere else for IllustriousXL character training.

### Adapter Type Comparison

**DoRA (Weight-Decomposed Low-Rank Adaptation)** — Winner
- Decomposes weight updates into learned magnitude vector + LoRA direction
- Peaks earlier than standard LoRA (e13 vs e17 with Prodigy)
- 100% finger/limb quality at peak vs 98% for standard LoRA
- ~82MB checkpoint at rank 32 (includes `dora_scale` magnitude vectors)
- Requires the 2D `dora_scale` fix for Forge/Reforge compatibility

**Standard LoRA** — Good fallback
- Works with both Prodigy and AdamW8bit
- Peaks later (e17 with Prodigy, e14 with AdamW8bit)
- Slightly lower ceiling than DoRA (~98% vs 100% hand quality)
- Smaller checkpoints (~65MB at rank 32)

**LoHa (Low-Rank Hadamard Adaptation)** — Situational
- Hadamard product of two low-rank factor pairs
- Better style separation in theory, but gradient starts near-zero
- **Incompatible with AdamW8bit** — needs Prodigy's adaptive 1e-3+ LR to bootstrap. At AdamW8bit's 2e-4, minimal imprinting even at epoch 8
- DoRA + LoHa is impossible (DoRA is `use_dora=True` on LoraConfig only, not LoHaConfig)

### Optimizer Comparison

**Prodigy** — Recommended
- Auto-adapting learning rate, no manual tuning
- `d_coef=0.4` — reined in for small character datasets (default 1.0 is too aggressive)
- `beta2=0.99` — **must use 0.99, not 0.999**. At 0.999, the second moment estimator averages over ~1000 steps of history. On small datasets (~5-10 steps/epoch), it never warms up and the LR stays near-zero. Nothing trains.
- Split UNet/TE instances are critical — a single Prodigy instance lets UNet dynamics dominate the adaptive LR estimate, giving TE an inappropriate learning rate
- Weight decay 0.01

**AdamW8bit** — Manual alternative
- Fixed LR: 2e-4 UNet, 2e-5 TE
- Peaks at e14 with DoRA (98% hands)
- Better hair color retention than Prodigy (stable LR = stable TE gradients)

### UNet Target Module Selection

**Q + K + V** (current) — character identity + color values
- `to_q`, `to_k`: Attention routing — which features to activate for the character concept
- `to_v`: Value projection — what color/texture those features carry
- `to_out.0` deliberately excluded — it causes style bleed from training data

**min-SNR buffers against V-induced overfitting** — V projections carry more style information than Q/K, but gamma=5 prevents the easy timesteps (where V matters most) from dominating.

### Cosine Schedule Tuning

Extending epochs from 18 to 22 improved the e15-20 quality band:
- At epoch 17 of 18: LR decayed to ~0.4% of peak. Nearly zero learning signal.
- At epoch 17 of 22: LR at ~1.5% of peak. 4x higher residual learning.
- The extra headroom means gentle refinement continues instead of memorizing gradient noise.

The cosine schedule's "tail" is where character LoRAs do their best work — identity is locked in, and the decaying LR polishes details without risking overfitting.

### Caption Processing for Characters

**Tag shuffling** with `keep_tokens=1`:
- Randomizes comma-separated tag order each time an image is seen
- Preserves trigger word in position 0 (first token)
- Prevents the model from memorizing "tag X always follows tag Y"
- Critical for generalization to novel prompt combinations

**Caption dropout disabled** (rate=0.0):
- Standard recommendation is 5% dropout for "better CFG quality"
- **Wrong for character LoRAs**: dropout trains the model to activate the character *unconditionally* (without any prompt). This means the character concept leaks through the CFG unconditional path, contaminating every generation even without the trigger word, but more importantly causes unavoidable style bleed.

**Trigger word injection**:
- Prepended automatically to every caption at runtime
- Not stored in caption files — dataset stays clean and reusable
- CONFIG is a frozen dataclass; trigger word injected via `Core.config.CONFIG = TrainingConfig(trigger_word=...)` before importing the trainer

### Base Model Selection

**Train on a model that matches your dataset's art style**, NOT the raw base model (Illustrious XL, NAI, etc.).

If your dataset is semi-realistic anime and you train on a flat-shading base, the learns the style gap between base and dataset as part of the charater.

### Checkpoint Format: Kohya-Compatible Safetensors

Checkpoints are saved in Kohya format for maximum compatibility:

- `lora_down.weight` — rank x in_features (low-rank factor A)
- `lora_up.weight` — out_features x rank (low-rank factor B)
- `.alpha` — scaling factor (float tensor, NOT optional)
- `dora_scale` — DoRA magnitude vector, **2D [N, 1]** (the critical fix)

**Alpha tensors are mandatory.** When missing, ComfyUI/A1111/Forge default to `alpha=rank`, giving 1.0 effective scale instead of the intended 0.5. This doubles the LoRA's strength silently, causing over-conditioning and detail loss.

**LoHa keys** use `hada_w1_a`/`hada_w1_b`/`hada_w2_a`/`hada_w2_b` format. PEFT appends `.default` adapter suffix that breaks naive key parsers — our backward-search parser finds known key names regardless of trailing suffixes.

### Per-Epoch Sample Generation

Two modes for visual progress tracking:

**Primary: Reforge/Forge API**
- Calls `/sdapi/v1/txt2img` with `<lora:name:strength>` prompt syntax
- Tests the LoRA on whatever model forge has loaded (can differ from training base)
- Auto-refreshes LoRA list via `/sdapi/v1/refresh-loras`
- Requires `--api` flag and `Outputs/` in Reforge's LoRA search paths
- Batch of 4 images assembled into 2x2 grid

**Fallback: Onboard Inference**
- Uses training model directly with EulerDiscreteScheduler
- 4 images with different seeds (seed, seed+1, seed+2, seed+3)
- Fixed seeds across epochs for direct visual comparison
- VAE temporarily brought back to GPU for decode, then re-offloaded

Samples saved to `Outputs/samples/sample_eNN.png`.

## Dataset Preparation

### Image Requirements
- Formats: `.png`, `.jpg`, `.webp`
- Any size — auto-resized to nearest bucket (1024x1024, 832x1216, or 1216x832)
- Recommended: at least 40 images of your character. Can be lower, but tested on 40. 
- Diverse poses, angles, expressions, and clothing

### Caption Format
- Matching `.txt` file per image with comma-separated Danbooru tags
- **Do NOT include the trigger word** — it's added automatically
- Do NOT include constant character features (see below)

### Character Tagging (Critical)

**Prune tags that describe constant character features:**
```
BEFORE (wrong):
1girl, solo, blue eyes, long hair, black hair, bangs, maid outfit, standing, smile

AFTER (correct):
1girl, solo, maid outfit, standing, smile
```

If your character always has blue eyes and black hair, **remove those tags**. Otherwise the model learns "blue eyes = only when tagged" instead of "blue eyes = this character."

**Keep tags that vary** between images: poses, expressions, clothing changes, backgrounds, composition (`solo`, `upper body`, `full body`).

**Keep `1girl`/`1boy`** — class tokens that help the model understand the subject type.

## Output

```
Outputs/
├── lora_epoch_01.safetensors  # ~82MB each (DoRA rank 32)
├── lora_epoch_02.safetensors
├── ...
├── lora_epoch_22.safetensors
└── samples/
    ├── sample_e01.png          # 2x2 grid per epoch
    ├── sample_e02.png
    └── ...
```

## System Requirements

- **GPU**: 16GB+ VRAM (RTX 3090, 4080, 4090, 5070 TI, etc.)
- **RAM**: 32GB+ recommended
- **CUDA**: 12.1+
- **Python**: 3.10+

Settings are optimized for 16GB VRAM. For less VRAM, reduce `batch_size` in [Core/config.py](Core/config.py).

## Troubleshooting

**"Out of memory"**
- Reduce `batch_size` to 2 or 1 in [Core/config.py](Core/config.py)
- Text encoders stay on GPU for runtime encoding (~1.6GB)

**"Trigger word doesn't activate the character"**
- Ensure `lora_text_encoder: True` (default)
- Use a more unique trigger word (no real English words or Danbooru tags)
- Check that you pruned constant features from captions (see Dataset Preparation)

**"Fingers/hands degraded"**
- Use an earlier epoch — peak hand quality is e13-17 depending on config
- Check TE alpha is rank/2 (not rank) — 1.0 scale melts details

**"Loss not decreasing"**
- Check captions are accurate and descriptive
- Try `optimizer = "adamw8bit"` for manual LR control
- If using Prodigy, ensure `prodigy_beta2 = 0.99` (not 0.999)

## Advanced Configuration

### Switching to AdamW8bit

```python
# In Core/config.py:
optimizer: str = "adamw8bit"
unet_learning_rate: float = 2e-4
text_encoder_learning_rate: float = 2e-5
```

Peaks at e14 with DoRA. More predictable convergence than Prodigy.

### Switching to Standard LoRA

```python
# In Core/config.py:
adapter_type: str = "lora"  # Instead of "dora"
```

Smaller checkpoints, slightly lower quality ceiling. Works with both optimizers.

### Switching to LoHa

```python
# In Core/config.py:
adapter_type: str = "loha"
optimizer: str = "prodigy"  # Required — AdamW8bit LR is too low for LoHa
```

Better style separation in theory. Requires Prodigy's adaptive LR to bootstrap.

### Connecting to Reforge/Forge API

1. Launch Reforge/Forge with `--api` flag
2. Add `Outputs/` to Reforge's LoRA search paths (or symlink)
3. Samples will auto-generate via API after each epoch checkpoint

```python
# In Core/config.py:
sample_api_url: str = "http://127.0.0.1:7860"  # Set to "" to disable API, use onboard only
sample_lora_strength: float = 1.0
```

## Bugs Fixed


1. **DoRA `dora_scale` dimensionality** — 1D vs 2D. Silent corruption or crash at inference.

2. **Missing `.alpha` tensors in checkpoints** — When absent, inference tools default to `alpha=rank` (1.0 scale) instead of the intended 0.5. Doubles effective LoRA strength silently.

3. **Text embeddings cached through base model** — Pre-caching text embeds bypasses TE LoRA entirely. The text encoder adapter never receives gradients. Trigger words don't train.

4. **TE1/TE2 gradient clipping independent** — Each clipped to norm 1.0 separately allows total TE gradient norm up to 2.0. Must clip together.

5. **LoHa checkpoint keys with PEFT `.default` suffix** — Naive key parsers skip all `hada_` weights, producing empty checkpoints.

6. **Prodigy with constant LR schedule** — Lambda=1.0 scheduler means Prodigy's auto-LR grows monotonically forever. Must apply cosine on top.

## License

MIT - do whatever you want with it.

---

No PhD required. No 47-tab GUI.
