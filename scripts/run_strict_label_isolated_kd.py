"""Plan or execute the label-isolated prediction-distillation experiment.

Dry-run is the default. The commands are printed unless ``--execute`` is
supplied.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


KD_GRID = "0,0.25,0.5,1,2,4,8,16,32,64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("citylens-population", "citylens-healthcare", "haidian", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=Path("strict_kd_runs"))
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--kd-weight-grid", default=KD_GRID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--citylens-population-features", type=Path)
    parser.add_argument("--citylens-population-reference-root", type=Path)
    parser.add_argument("--citylens-healthcare-features", type=Path)
    parser.add_argument("--citylens-healthcare-reference-root", type=Path)
    parser.add_argument("--haidian-features", type=Path)
    return parser.parse_args()


def require(path: Path | None, option: str, execute: bool) -> Path:
    if path is None:
        raise ValueError(f"{option} is required for the selected profile")
    resolved = path.expanduser().resolve()
    if execute and not resolved.exists():
        raise FileNotFoundError(f"{option} does not exist: {resolved}")
    return resolved


def run(command: list[str], marker: Path, args: argparse.Namespace, cwd: Path) -> None:
    if args.skip_existing and marker.is_file() and marker.stat().st_size > 0:
        print(f"SKIP\t{marker}")
        return
    print(shlex.join(command))
    if args.execute:
        marker.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=cwd, check=True)


def runtime(args: argparse.Namespace) -> list[str]:
    values = ["--device", args.device, "--strict-label-isolation"]
    if args.no_progress:
        values.append("--no-progress")
    return values


def citylens(
    task: str,
    features: Path,
    reference_root: Path,
    seeds: list[int],
    args: argparse.Namespace,
    script_dir: Path,
    output_root: Path,
) -> None:
    prefix = f"citylens_{task}"
    for seed in seeds:
        # Population's published KD sweep fixed the seed-42 reference, whereas
        # Healthcare used the corresponding per-seed all-label reference.
        reference_seed = 42 if task == "population" else seed
        reference = reference_root / (
            f"citylens_{task}_within_city_seed{reference_seed}"
            if task == "population"
            else f"citylens_healthcare_all_within_city_seed{reference_seed}"
        )
        output = output_root / f"{prefix}_prediction_kd_controls_grid64_strictiso_seed{seed}"
        command = [
            args.python,
            str(script_dir / "run_citylens_prediction_kd_controls.py"),
            "--features", str(features),
            "--reference-dir", str(reference),
            "--output-dir", str(output),
            "--target-transform", "log1p",
            "--kd-weight-grid", args.kd_weight_grid,
            "--spatial-seed", "42",
            "--shuffle-seed", "42",
            "--seed", str(seed),
            *runtime(args),
        ]
        run(command, output / "pooled_summary.csv", args, script_dir)


def haidian(
    features: Path,
    seeds: list[int],
    args: argparse.Namespace,
    script_dir: Path,
    output_root: Path,
) -> None:
    for seed in seeds:
        output = output_root / (
            f"haidian_population_nested_spatial_localshuffle_grid64_strictiso_seed{seed}"
        )
        command = [
            args.python,
            str(script_dir / "run_haidian_nested_spatial_controls.py"),
            "--features", str(features),
            "--output-dir", str(output),
            "--targets", "population",
            "--outer-folds", "5",
            "--inner-folds", "4",
            "--spatial-blocks", "25",
            "--kd-weight-grid", args.kd_weight_grid,
            "--lambda-pred", "0.5",
            "--spatial-seed", "42",
            "--shuffle-seed", "42",
            "--seed", str(seed),
            *runtime(args),
        ]
        run(command, output / "pooled_summary.csv", args, script_dir)


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds is empty")
    script_dir = Path(__file__).resolve().parent
    output_root = args.output_root.expanduser().resolve()
    print("EXECUTE" if args.execute else "DRY RUN")
    print(f"profile={args.profile} seeds={seeds} output={output_root}")

    if args.profile in {"citylens-population", "all"}:
        citylens(
            "population",
            require(args.citylens_population_features, "--citylens-population-features", args.execute),
            require(args.citylens_population_reference_root, "--citylens-population-reference-root", args.execute),
            seeds, args, script_dir, output_root,
        )
    if args.profile in {"citylens-healthcare", "all"}:
        citylens(
            "healthcare_all",
            require(args.citylens_healthcare_features, "--citylens-healthcare-features", args.execute),
            require(args.citylens_healthcare_reference_root, "--citylens-healthcare-reference-root", args.execute),
            seeds, args, script_dir, output_root,
        )
    if args.profile in {"haidian", "all"}:
        haidian(
            require(args.haidian_features, "--haidian-features", args.execute),
            seeds, args, script_dir, output_root,
        )
    if not args.execute:
        print("\nNo training was run. Add --execute after reviewing the commands.")


if __name__ == "__main__":
    main()
