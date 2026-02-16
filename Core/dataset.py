"""
EasyDora Dataset
Dataset loader for character LoRA training with tag shuffling and caption dropout.
Caches VAE latents only - text encoding happens at runtime through LoRA text encoders.
"""
import hashlib
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import CONFIG

# Valid training resolutions (width, height)
VALID_RESOLUTIONS = [
    (1024, 1024),  # Square
    (832, 1216),   # Portrait
    (1216, 832),   # Landscape
]


def get_target_resolution(width: int, height: int) -> tuple[int, int]:
    """Determine best target resolution based on aspect ratio."""
    aspect = width / height

    # Find closest matching resolution by aspect ratio
    best_res = None
    best_diff = float('inf')

    for w, h in VALID_RESOLUTIONS:
        target_aspect = w / h
        diff = abs(aspect - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best_res = (w, h)

    return best_res


def resize_to_bucket(image: Image.Image) -> Image.Image:
    """Resize image to nearest valid resolution bucket if needed."""
    width, height = image.size

    # Check if already valid
    if (width, height) in VALID_RESOLUTIONS:
        return image

    # Get target resolution
    target_w, target_h = get_target_resolution(width, height)

    return image.resize((target_w, target_h), Image.LANCZOS)


class AspectRatioBucketSampler:
    """
    Batch sampler that groups images by aspect ratio bucket.
    Ensures each batch contains images of the same size.
    Yields batches of indices (not individual indices).
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by aspect ratio bucket
        self.buckets = defaultdict(list)
        for idx, (img_path, _) in enumerate(dataset.samples):
            # Get dimensions from cache or image
            cache_path = dataset._get_cache_path(img_path)
            if cache_path.exists():
                cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
                size = cache_data["original_size"]
            else:
                with Image.open(img_path) as img:
                    # Determine target bucket (will be resized during caching)
                    target_w, target_h = get_target_resolution(img.width, img.height)
                    size = (target_h, target_w)  # (height, width) format

            # Create bucket key from dimensions
            bucket_key = size
            self.buckets[bucket_key].append(idx)

        # Pre-compute all batches
        self._create_batches()

        print(f"  Aspect ratio buckets: {len(self.buckets)}")
        for size, indices in self.buckets.items():
            total_with_repeats = len(indices) * CONFIG.num_repeats
            print(f"    {size[1]}x{size[0]}: {len(indices)} images ({total_with_repeats} with {CONFIG.num_repeats}x repeats)")

    def _create_batches(self):
        """Create all batches from buckets."""
        self.batches = []

        for bucket_key, base_indices in self.buckets.items():
            # Multiply indices by num_repeats to account for dataset repetition
            bucket_indices = []
            for repeat in range(CONFIG.num_repeats):
                # Add offset for each repeat cycle
                offset = repeat * len(self.dataset.samples)
                bucket_indices.extend([idx + offset for idx in base_indices])

            if self.shuffle:
                random.shuffle(bucket_indices)

            # Create batches from this bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                self.batches.append(batch)

        # Shuffle batches across buckets
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        # Recreate batches each epoch for different shuffling
        self._create_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class ImageCaptionDataset(Dataset):
    """
    Dataset for character LoRA training.
    Caches VAE latents only. Captions are processed at runtime with
    trigger word prepending, tag shuffling, and caption dropout.
    """

    def __init__(self, vae, device: str = "cuda"):
        self.dataset_dir = Path(CONFIG.dataset_dir)
        self.cache_dir = Path(CONFIG.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.vae = vae
        self.device = device

        # Find all image-caption pairs
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} image-caption pairs")

        # Pre-load all captions into memory (tiny - a few KB total)
        self.captions = {}
        for img_path, txt_path in self.samples:
            self.captions[img_path] = txt_path.read_text(encoding="utf-8").strip()

        # Pre-compute and cache VAE latents
        self._cache_latents()

    def _find_samples(self) -> list[tuple[Path, Path]]:
        """Find all image.png + image.txt pairs."""
        samples = []
        for img_path in self.dataset_dir.glob("*.png"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                samples.append((img_path, txt_path))

        # Also check for .jpg and .webp
        for ext in [".jpg", ".jpeg", ".webp"]:
            for img_path in self.dataset_dir.glob(f"*{ext}"):
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    samples.append((img_path, txt_path))

        return sorted(samples, key=lambda x: x[0].name)

    def _get_cache_path(self, img_path: Path) -> Path:
        """Generate cache path for an image based on file hash."""
        # Hash based on filename + modification time for cache invalidation
        key = f"{img_path.name}_{img_path.stat().st_mtime}"
        hash_id = hashlib.md5(key.encode()).hexdigest()[:12]
        # "_lat_" prefix distinguishes from old format that included text embeddings
        return self.cache_dir / f"{img_path.stem}_lat_{hash_id}.pt"

    def _cache_latents(self):
        """Pre-compute VAE latents for all samples (text encoding happens at runtime)."""
        print("Caching VAE latents (this speeds up future training runs)...")

        uncached = [
            (img, txt) for img, txt in self.samples
            if not self._get_cache_path(img).exists()
        ]

        if not uncached:
            print("All latents already cached!")
            return

        print(f"Computing latents for {len(uncached)} images...")

        # VAE encoding must be done in float32 for numerical stability
        original_vae_dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)

        with torch.no_grad():
            for img_path, txt_path in tqdm(uncached, desc="Caching"):
                cache_path = self._get_cache_path(img_path)

                # Load image, strip all metadata (ICC profiles, gamma, EXIF)
                raw_image = Image.open(img_path)
                image = Image.fromarray(np.array(raw_image.convert("RGB")))
                raw_image.close()

                # Resize to valid bucket if needed
                image = resize_to_bucket(image)

                # Get image dimensions for aspect ratio bucket
                width, height = image.size

                # Convert to tensor via numpy (fast, no Python pixel loop)
                img_array = np.array(image, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
                img_tensor = (img_tensor * 2.0 - 1.0).unsqueeze(0).to(
                    device=self.device, dtype=torch.float32
                )

                # Compute VAE latents in float32 for numerical stability
                latents = self.vae.encode(img_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                # Cast to bf16 for storage (training will use bf16)
                latents = latents.to(dtype=torch.bfloat16)

                # Save latents only (text encoding happens at runtime)
                cache_data = {
                    "latents": latents.cpu(),
                    "original_size": (height, width),
                    "crop_coords": (0, 0),
                    "target_size": (height, width),
                }
                torch.save(cache_data, cache_path)

        # Restore VAE to original dtype
        self.vae.to(dtype=original_vae_dtype)

        print("Latent caching complete!")

    def _process_caption(self, caption: str) -> str:
        """Apply character LoRA caption processing: trigger word, shuffle, dropout."""
        # Caption dropout - return empty string with probability
        if CONFIG.caption_dropout_rate > 0 and random.random() < CONFIG.caption_dropout_rate:
            return ""

        # Split into comma-separated tags
        tags = [tag.strip() for tag in caption.split(",") if tag.strip()]

        # Prepend trigger word
        if CONFIG.trigger_word:
            tags.insert(0, CONFIG.trigger_word)

        # Shuffle tags (preserving first keep_tokens)
        if CONFIG.shuffle_tags and len(tags) > CONFIG.keep_tokens:
            preserved = tags[:CONFIG.keep_tokens]
            shuffleable = tags[CONFIG.keep_tokens:]
            random.shuffle(shuffleable)
            tags = preserved + shuffleable

        return ", ".join(tags)

    def __len__(self) -> int:
        return len(self.samples) * CONFIG.num_repeats

    def __getitem__(self, idx: int) -> dict:
        # Cycle through samples with repeats
        actual_idx = idx % len(self.samples)
        img_path, _ = self.samples[actual_idx]
        cache_path = self._get_cache_path(img_path)

        # Load VAE latents from cache
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)

        # Process caption at runtime (trigger word + shuffle + dropout)
        caption = self.captions[img_path]
        processed_caption = self._process_caption(caption)

        return {
            "latents": cache_data["latents"].squeeze(0),
            "caption": processed_caption,
            "original_size": cache_data["original_size"],
            "crop_coords": cache_data["crop_coords"],
            "target_size": cache_data["target_size"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for batches (same aspect ratio guaranteed by sampler)."""
    return {
        "latents": torch.stack([b["latents"] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "original_sizes": [b["original_size"] for b in batch],
        "crop_coords": [b["crop_coords"] for b in batch],
        "target_sizes": [b["target_size"] for b in batch],
    }
