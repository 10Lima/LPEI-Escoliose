import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUBSET5_DIR = ROOT / "raw" / "images" / "test" / "Spinal-AI2024-subset5"
GT_CSV = ROOT / "processed" / "cleaned" / "cobb_test_gt_subset5.csv"
CHECKPOINT205_METRICS = ROOT / "results" / "final" / "checkpoint205" / "checkpoint205_metrics_summary.csv"


def read_metrics_row(path, stage_name):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    for row in rows:
        if row.get("stage") == stage_name or row.get("scenario") == stage_name:
            return row
    raise ValueError(f"Row '{stage_name}' not found in: {path}")


def smoke(num_images):
    if not SUBSET5_DIR.exists():
        raise FileNotFoundError(f"Subset5 not found: {SUBSET5_DIR}")
    if not GT_CSV.exists():
        raise FileNotFoundError(f"Ground truth not found: {GT_CSV}")
    if not CHECKPOINT205_METRICS.exists():
        raise FileNotFoundError(f"Checkpoint205 metrics not found: {CHECKPOINT205_METRICS}")

    images = sorted(SUBSET5_DIR.glob("*.jpg"))
    with GT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    if len(images) != 4000:
        raise ValueError(f"Expected 4000 subset5 images; found {len(images)}")
    if len(gt_rows) != 4000:
        raise ValueError(f"Expected 4000 GT rows; found {len(gt_rows)}")

    gt_by_image = {row["image"]: row for row in gt_rows}
    missing_gt = [image.name for image in images if image.name not in gt_by_image]
    if missing_gt:
        raise ValueError(f"Images without GT: {missing_gt[:5]}")

    metrics = read_metrics_row(CHECKPOINT205_METRICS, "checkpoint205")
    sample = images[:num_images]

    print("Smoke check OK")
    print(f"Subset5: {len(images)} images")
    print(f"GT: {len(gt_rows)} rows")
    print(f"Sample checked: {len(sample)} images")
    print("\nCheckpoint205 reference:")
    print(f"  MAE3: {metrics.get('mae3')}")
    print(f"  RMSE3: {metrics.get('rmse3')}")
    print(f"  <=5 degrees: {metrics.get('within_5')}%")
    print(f"  failures >5: {metrics.get('failures_gt5')}")
    print(f"  severe failures >8: {metrics.get('severe_gt8')}")


def main():
    parser = argparse.ArgumentParser(description="Smoke checks for the LPEI scoliosis evaluation pack.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Validate structure, subset5, GT, and reference metrics.")
    smoke_parser.add_argument("--num-images", type=int, default=8)

    args = parser.parse_args()
    if args.command == "smoke":
        smoke(max(args.num_images, 0))


if __name__ == "__main__":
    main()
