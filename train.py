#!/usr/bin/env python3
"""
EasyDora - IllustriousXL Character LoRA Trainer

No GUI, no config files - just specify a trigger word and train.

Usage:
    python train.py mycharname          # Trigger word from CLI
    python train.py                     # Reads from Dataset/trigger_word.txt or prompts

Ensure:
    - Base model (.safetensors) is in ./BaseModel/
    - Dataset images (.png) and captions (.txt) are in ./Dataset/
    - LoRA checkpoints will be saved to ./Outputs/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_model_exists():
    """Check if a model file exists in BaseModel/."""
    base_model_dir = Path("BaseModel")

    # Check for safetensors file
    if list(base_model_dir.glob("*.safetensors")):
        return True

    # Check for diffusers format (legacy support)
    required_dirs = ["unet", "vae", "text_encoder", "text_encoder_2"]
    if all((base_model_dir / d).exists() for d in required_dirs):
        return True

    print("Error: No model found in BaseModel/")
    print("Please add a .safetensors file (e.g., illustrious-xl.safetensors)")
    return False


def get_trigger_word() -> str:
    """Get trigger word from CLI arg, file, or interactive prompt."""
    # Check CLI argument first: python train.py mychar
    if len(sys.argv) > 1:
        word = sys.argv[1].strip()
        if word:
            return word

    # Check for trigger_word.txt in dataset dir
    trigger_file = Path("Dataset") / "trigger_word.txt"
    if trigger_file.exists():
        word = trigger_file.read_text(encoding="utf-8").strip()
        if word:
            print(f"Trigger word from file: '{word}'")
            return word

    # Interactive prompt
    print("Character LoRA training requires a trigger word.")
    print("This unique token activates your character at inference time.")
    print("Use something rare/unique (e.g., chr_mychar, sks_name, r0gue)")
    print()
    word = input("Enter trigger word: ").strip()
    if not word:
        print("Error: Trigger word cannot be empty for character LoRA training.")
        sys.exit(1)

    if " " in word:
        print(f"Warning: '{word}' contains spaces. Consider using underscores instead.")

    return word


if __name__ == "__main__":
    print()
    print("+" + "=" * 58 + "+")
    print("|          EasyDora - IllustriousXL Character LoRA          |")
    print("|       DoRA + Prodigy | Tag Shuffle | Just Works          |")
    print("+" + "=" * 58 + "+")
    print()

    if not check_model_exists():
        sys.exit(1)

    # Get trigger word and inject into config
    trigger_word = get_trigger_word()
    print(f"Trigger word: '{trigger_word}'")
    print()

    from Core.config import TrainingConfig
    import Core.config
    Core.config.CONFIG = TrainingConfig(trigger_word=trigger_word)

    from Core.trainer import run_training
    run_training()
