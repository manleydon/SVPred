"""Extract paired satellite and real-street DINOv2 features.

This script only creates a reusable feature archive. It does not train a
downstream model. Multiple street-view filenames separated by semicolons are
supported and are mean-pooled after image-level feature extraction.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


TARGET_COLUMNS = ("population", "log_Carbon", "BuildingHeight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Paired-region CSV with one row per satellite tile.",
    )
    parser.add_argument(
        "--satellite-dir",
        type=Path,
        required=True,
        help="Directory holding the satellite tiles named in the CSV.",
    )
    parser.add_argument(
        "--street-dir",
        type=Path,
        required=True,
        help="Directory holding the street-view images named in the CSV.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="./dinov2-base",
        help=(
            "Local Hugging Face DINOv2 directory. If the directory does not "
            "exist, --hub-model is loaded with torch.hub."
        ),
    )
    parser.add_argument("--hub-model", default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_transform(image_size: int) -> transforms.Compose:
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


def parse_street_files(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def image_decode_error(path: Path) -> str | None:
    """Return a concise error if PIL cannot completely decode an image."""
    try:
        # verify() checks the container/checksum without retaining decoded data.
        with Image.open(path) as image:
            image.verify()
        # Reopen because verify() invalidates the decoder, then force complete
        # RGB decoding. Some truncated PNGs fail only when pixel data is read.
        with Image.open(path) as image:
            image.convert("RGB").load()
    except (OSError, SyntaxError, ValueError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def validate_and_filter_rows(
    csv_path: Path,
    satellite_dir: Path,
    street_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {
        "wgs84_lat",
        "wgs84_lng",
        "Filename",
        "streetview_files",
        *TARGET_COLUMNS,
    }
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    keep_rows: list[int] = []
    invalid_rows: list[dict[str, object]] = []

    for row_index, row in df.iterrows():
        satellite_path = satellite_dir / str(row["Filename"]).strip()
        street_files = parse_street_files(row["streetview_files"])
        street_paths = [street_dir / filename for filename in street_files]
        problems: list[tuple[str, Path, str]] = []
        candidates = [("satellite", satellite_path)] + [
            ("street", path) for path in street_paths
        ]
        if not street_paths:
            problems.append(("street", street_dir, "no street-view filename"))
        for modality, path in candidates:
            if not path.is_file():
                problems.append((modality, path, "file not found"))
            else:
                decode_error = image_decode_error(path)
                if decode_error is not None:
                    problems.append((modality, path, decode_error))

        if problems:
            for modality, path, reason in problems:
                invalid_rows.append(
                    {
                        "row_index": int(row_index),
                        "sample_id": str(row["Filename"]),
                        "modality": modality,
                        "path": str(path),
                        "reason": reason,
                    }
                )
        else:
            keep_rows.append(row_index)

    invalid = pd.DataFrame(
        invalid_rows,
        columns=("row_index", "sample_id", "modality", "path", "reason"),
    )
    if not invalid.empty:
        preview = ", ".join(invalid["sample_id"].drop_duplicates().head(5))
        excluded_count = invalid["row_index"].nunique()
        print(
            f"Warning: excluded {excluded_count} rows with missing or corrupt images. "
            f"Examples: {preview}"
        )

    filtered = df.loc[keep_rows].copy().reset_index(drop=True)
    if filtered.empty:
        raise RuntimeError("No valid satellite/street-view pairs were found.")
    return filtered, invalid


class PairedImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        satellite_dir: Path,
        street_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.satellite_dir = satellite_dir
        self.street_dir = street_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        row = self.dataframe.iloc[index]
        satellite_path = self.satellite_dir / str(row["Filename"]).strip()
        satellite = self.transform(Image.open(satellite_path).convert("RGB"))

        street_images = []
        for filename in parse_street_files(row["streetview_files"]):
            street_path = self.street_dir / filename
            street_images.append(self.transform(Image.open(street_path).convert("RGB")))
        return satellite, street_images


def collate_pairs(
    batch: Sequence[tuple[torch.Tensor, list[torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    satellite_batch = torch.stack([item[0] for item in batch], dim=0)
    counts = [len(item[1]) for item in batch]
    flat_street_batch = torch.stack(
        [street for _, street_list in batch for street in street_list], dim=0
    )
    return satellite_batch, flat_street_batch, counts


def load_encoder(local_model: str, hub_model: str) -> torch.nn.Module:
    if os.path.isdir(local_model):
        from transformers import AutoModel

        print(f"Loading local Hugging Face model: {local_model}")
        model = AutoModel.from_pretrained(local_model)
    else:
        print(
            f"Local model directory not found: {local_model}. "
            f"Loading facebookresearch/dinov2:{hub_model} with torch.hub."
        )
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
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    satellite_features: list[np.ndarray] = []
    street_features: list[np.ndarray] = []

    for satellite, flat_street, counts in tqdm(dataloader, desc="DINOv2 features"):
        satellite_embedding = cls_token(model, satellite.to(device))
        flat_street_embedding = cls_token(model, flat_street.to(device))

        pooled_street = []
        offset = 0
        for count in counts:
            pooled_street.append(flat_street_embedding[offset : offset + count].mean(dim=0))
            offset += count

        satellite_features.append(satellite_embedding.cpu().float().numpy())
        street_features.append(torch.stack(pooled_street).cpu().float().numpy())

    return (
        np.concatenate(satellite_features, axis=0),
        np.concatenate(street_features, axis=0),
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataframe, invalid_images = validate_and_filter_rows(
        args.csv,
        args.satellite_dir,
        args.street_dir,
    )
    print(f"Valid paired samples: {len(dataframe)}")

    dataset = PairedImageDataset(
        dataframe=dataframe,
        satellite_dir=args.satellite_dir,
        street_dir=args.street_dir,
        transform=get_transform(args.image_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_pairs,
        pin_memory=device.type == "cuda",
    )

    model = load_encoder(args.model, args.hub_model).to(device)
    satellite_features, street_features = extract_features(model, dataloader, device)

    targets = dataframe.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not invalid_images.empty:
        invalid_manifest = args.output.with_name(
            f"{args.output.stem}_invalid_images.csv"
        )
        invalid_images.to_csv(invalid_manifest, index=False)
        print(f"Saved invalid-image manifest: {invalid_manifest}")
    # Force metadata columns to fixed-width Unicode arrays. Pandas otherwise
    # commonly exports them as dtype=object, which requires pickle to reload.
    sample_ids = np.asarray(dataframe["Filename"].astype(str).tolist(), dtype=np.str_)
    satellite_files = np.asarray(
        dataframe["Filename"].astype(str).tolist(), dtype=np.str_
    )
    street_files = np.asarray(
        dataframe["streetview_files"].astype(str).tolist(), dtype=np.str_
    )
    np.savez_compressed(
        args.output,
        sample_id=sample_ids,
        latitude=dataframe["wgs84_lat"].to_numpy(dtype=np.float64),
        longitude=dataframe["wgs84_lng"].to_numpy(dtype=np.float64),
        satellite_features=satellite_features.astype(np.float32),
        street_features=street_features.astype(np.float32),
        targets=targets,
        target_names=np.asarray(TARGET_COLUMNS, dtype=np.str_),
        satellite_files=satellite_files,
        street_files=street_files,
        encoder=np.asarray([args.hub_model], dtype=np.str_),
    )
    print(f"Saved feature archive: {args.output}")
    print(f"Satellite features: {satellite_features.shape}")
    print(f"Street features: {street_features.shape}")


if __name__ == "__main__":
    main()
