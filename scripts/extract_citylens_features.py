"""Extract frozen DINOv2 features for paired CityLens regions.

The task JSON supplies area identifiers and labels through its ``reference``
field. Local Mapillary files are resolved by area under
``street_view_image/Mapillary/image/<area>/``; up to ten selected views are
mean-pooled after encoding. Missing or undecodable inputs exclude the whole
region, and selection does not backfill a failed decode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root of an authorized local CityLens release.",
    )
    parser.add_argument(
        "--task-json",
        type=Path,
        default=Path("Dataset/all_global_build_height_task-all.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("citylens_buildheight_dinov2b14_mapillary.npz"),
    )
    parser.add_argument(
        "--target-name",
        default="BuildingHeight",
        help=(
            "Metadata name stored in the output; labels are read from the "
            "task JSON reference field."
        ),
    )
    parser.add_argument("--model", default="./dinov2-base")
    parser.add_argument("--hub-model", default="dinov2_vitb14")
    parser.add_argument("--max-street-images", type=int, default=10)
    parser.add_argument("--min-street-images", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--region-batch-size",
        type=int,
        default=8,
        help="Each region adds up to 10 street images to the GPU batch.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def city_from_json_item(item: dict[str, object]) -> str:
    images = item.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Task item has no images list.")
    return PurePosixPath(str(images[0])).parent.name


def image_decode_error(path: Path) -> str | None:
    """Return a concise error if an image cannot be completely decoded."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
    except (OSError, SyntaxError, ValueError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def parse_tile(area: str) -> tuple[int, int]:
    parts = area.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected CityLens area id: {area}")
    return int(parts[0]), int(parts[1])


def build_manifest(args: argparse.Namespace) -> list[dict[str, object]]:
    task_path = args.task_json
    if not task_path.is_absolute():
        task_path = args.data_root / task_path
    with task_path.open("r", encoding="utf-8") as handle:
        task_items = json.load(handle)
    if not isinstance(task_items, list):
        raise ValueError("Expected the CityLens task JSON to contain a list.")

    manifest: list[dict[str, object]] = []
    missing_satellite = 0
    missing_street = 0
    invalid_image_rows = 0
    for item in task_items:
        area = str(item["area"])
        city = city_from_json_item(item)
        tile_x, tile_y = parse_tile(area)
        satellite_path = args.data_root / "satellite_image" / city / f"{area}.png"
        street_directory = (
            args.data_root / "street_view_image" / "Mapillary" / "image" / area
        )
        street_paths = []
        if street_directory.is_dir():
            street_paths = sorted(
                path
                for path in street_directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )[: args.max_street_images]

        if not satellite_path.is_file():
            missing_satellite += 1
            continue
        if len(street_paths) < args.min_street_images:
            missing_street += 1
            continue
        invalid_paths = []
        for path in [satellite_path, *street_paths]:
            decode_error = image_decode_error(path)
            if decode_error is not None:
                invalid_paths.append((path, decode_error))
        if invalid_paths:
            invalid_image_rows += 1
            examples = "; ".join(
                f"{path.name}: {reason}" for path, reason in invalid_paths[:2]
            )
            print(f"Excluded undecodable region {area}: {examples}")
            continue
        manifest.append(
            {
                "area": area,
                "city": city,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "target": float(item["reference"]),
                "satellite_path": satellite_path,
                "street_paths": street_paths,
            }
        )

    print(
        f"Task rows={len(task_items)} | paired={len(manifest)} | "
        f"missing satellite={missing_satellite} | missing Mapillary={missing_street} | "
        f"undecodable image rows={invalid_image_rows}"
    )
    city_counts: dict[str, int] = {}
    for row in manifest:
        city = str(row["city"])
        city_counts[city] = city_counts.get(city, 0) + 1
    for city, count in sorted(city_counts.items()):
        print(f"  {city:14s} {count:4d} paired regions")
    if not manifest:
        raise RuntimeError("No paired CityLens regions found.")
    return manifest


class CityLensDataset(Dataset):
    def __init__(
        self,
        manifest: list[dict[str, object]],
        transform: transforms.Compose,
    ) -> None:
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        row = self.manifest[index]
        satellite = self.transform(
            Image.open(Path(row["satellite_path"])).convert("RGB")
        )
        streets = [
            self.transform(Image.open(path).convert("RGB"))
            for path in row["street_paths"]
        ]
        return satellite, streets


def collate_regions(
    batch: Sequence[tuple[torch.Tensor, list[torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    satellite = torch.stack([item[0] for item in batch])
    counts = [len(item[1]) for item in batch]
    flat_streets = torch.stack(
        [street for _, street_list in batch for street in street_list]
    )
    return satellite, flat_streets, counts


def load_encoder(local_model: str, hub_model: str) -> torch.nn.Module:
    if os.path.isdir(local_model):
        from transformers import AutoModel

        print(f"Loading local Hugging Face model: {local_model}")
        model = AutoModel.from_pretrained(local_model)
    else:
        print(f"Loading torch.hub model: facebookresearch/dinov2:{hub_model}")
        model = torch.hub.load("facebookresearch/dinov2", hub_model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def cls_token(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported encoder output type: {type(output)!r}")


@torch.inference_mode()
def extract(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    satellite_features: list[np.ndarray] = []
    street_features: list[np.ndarray] = []
    for satellite, flat_streets, counts in tqdm(dataloader, desc="CityLens DINOv2"):
        satellite_embedding = cls_token(model, satellite.to(device))
        street_embedding = cls_token(model, flat_streets.to(device))
        pooled = []
        offset = 0
        for count in counts:
            pooled.append(street_embedding[offset : offset + count].mean(dim=0))
            offset += count
        satellite_features.append(satellite_embedding.cpu().float().numpy())
        street_features.append(torch.stack(pooled).cpu().float().numpy())
    return np.concatenate(satellite_features), np.concatenate(street_features)


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    device = resolve_device(args.device)
    dataset = CityLensDataset(manifest, image_transform(args.image_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.region_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        collate_fn=collate_regions,
    )
    model = load_encoder(args.model, args.hub_model).to(device)
    satellite_features, street_features = extract(model, dataloader, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_id=np.asarray([row["area"] for row in manifest], dtype=np.str_),
        city=np.asarray([row["city"] for row in manifest], dtype=np.str_),
        tile_x=np.asarray([row["tile_x"] for row in manifest], dtype=np.int64),
        tile_y=np.asarray([row["tile_y"] for row in manifest], dtype=np.int64),
        target=np.asarray([row["target"] for row in manifest], dtype=np.float32),
        satellite_features=satellite_features.astype(np.float32),
        street_features=street_features.astype(np.float32),
        street_count=np.asarray(
            [len(row["street_paths"]) for row in manifest], dtype=np.int16
        ),
        target_name=np.asarray([args.target_name], dtype=np.str_),
        street_source=np.asarray(["Mapillary"], dtype=np.str_),
        encoder=np.asarray([args.hub_model], dtype=np.str_),
    )
    print(f"Saved: {args.output}")
    print(f"Satellite features: {satellite_features.shape}")
    print(f"Street features: {street_features.shape}")


if __name__ == "__main__":
    main()
