import csv
import os
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import (
    TARGETS,
    metric_summary_from_arrays,
    print_required_metric_block,
    round_safe,
    safe_float,
)


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT186_DIR = Path(os.environ.get(
    "SPINAL_191_CHECKPOINT186_DIR",
    str(ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1"),
))
MT190_DIR = Path(os.environ.get(
    "SPINAL_191_MT190_DIR",
    str(ACTIVE_DIR / "validar_mt188_protegido_checkpoint186_v3"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_191_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "checkpoint191_checkpoint186_mt190_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_dir_has_results(output_dir):
    if not output_dir.exists():
        return False
    for item in output_dir.iterdir():
        if item.is_file():
            return True
        if item.is_dir() and any(item.iterdir()):
            return True
    return False


def prepare_output_dir():
    if output_dir_has_results(OUTPUT_DIR):
        raise FileExistsError(f"Pasta de output ja tem ficheiros: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def target_index(rows):
    indexed = {}
    for row in rows:
        indexed.setdefault(row["file_id"], {})[row["target"]] = row
    return indexed


def arrays_from_target_rows(target_rows):
    by_file = target_index(target_rows)
    preds = []
    gts = []
    file_ids = sorted(by_file)
    for file_id in file_ids:
        preds.append([safe_float(by_file[file_id][target]["estimated"]) for target in TARGETS])
        gts.append([safe_float(by_file[file_id][target]["gt"]) for target in TARGETS])
    return file_ids, np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)


def build_checkpoint_rows(mt190_target_rows):
    target_rows = []
    for row in mt190_target_rows:
        target_rows.append({
            "file_id": row["file_id"],
            "raw_file_id": row.get("raw_file_id", ""),
            "target": row["target"],
            "gt": row["gt"],
            "estimated": row["estimated"],
            "abs_error": row["abs_error"],
            "source191": row["source190"] if row["source190"] != "checkpoint186" else "checkpoint186",
            "source190": row["source190"],
        })

    by_file = target_index(target_rows)
    image_rows = []
    for file_id in sorted(by_file):
        values = by_file[file_id]
        errors = [safe_float(values[target]["abs_error"]) for target in TARGETS]
        changed = [target for target in TARGETS if values[target]["source191"] == "adaptive190_MT"]
        image_rows.append({
            "file_id": file_id,
            "raw_file_id": values["PT"].get("raw_file_id", ""),
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(float(np.mean(errors))),
            "within_5": int(float(np.mean(errors)) <= 5.0),
            "pt_source191": values["PT"]["source191"],
            "mt_source191": values["MT"]["source191"],
            "tll_source191": values["TL_L"]["source191"],
            "changed_targets191": "+".join(changed),
        })
    return target_rows, image_rows


def validate_against_mt190(checkpoint_target_rows, checkpoint_image_rows):
    mt190_target_rows = read_csv(MT190_DIR / "mt190_best_safe_target_rows.csv")
    mt190_image_rows = read_csv(MT190_DIR / "mt190_best_safe_image_rows.csv")
    own_targets = target_index(checkpoint_target_rows)
    ref_targets = target_index(mt190_target_rows)
    if set(own_targets) != set(ref_targets):
        raise RuntimeError("Ficheiros alvo divergem entre checkpoint191 e mt190.")
    for file_id in own_targets:
        for target in TARGETS:
            own = own_targets[file_id][target]
            ref = ref_targets[file_id][target]
            for key in ["gt", "estimated", "abs_error"]:
                if abs(safe_float(own[key]) - safe_float(ref[key])) > 1e-4:
                    raise RuntimeError(f"Divergencia em {file_id} {target} {key}")

    own_images = {row["file_id"]: row for row in checkpoint_image_rows}
    ref_images = {row["file_id"]: row for row in mt190_image_rows}
    if set(own_images) != set(ref_images):
        raise RuntimeError("Ficheiros imagem divergem entre checkpoint191 e mt190.")
    for file_id, own in own_images.items():
        ref = ref_images[file_id]
        for own_key, ref_key in [
            ("pt_abs_error", "pt_abs_error"),
            ("mt_abs_error", "mt_abs_error"),
            ("tll_abs_error", "tll_abs_error"),
            ("mae3", "mae3"),
        ]:
            if abs(safe_float(own[own_key]) - safe_float(ref[ref_key])) > 1e-4:
                raise RuntimeError(f"Divergencia imagem {file_id} {own_key}")


def metric_rows(checkpoint186_target_rows, checkpoint191_target_rows):
    _, base_preds, gts = arrays_from_target_rows(checkpoint186_target_rows)
    _, preds, gts191 = arrays_from_target_rows(checkpoint191_target_rows)
    if not np.allclose(gts, gts191):
        raise RuntimeError("GT divergente entre checkpoint186 e checkpoint191.")
    base_summary = metric_summary_from_arrays(base_preds, gts)
    final_summary = metric_summary_from_arrays(preds, gts, baseline_preds=base_preds)
    return [
        {"stage": "checkpoint186", **base_summary},
        {"stage": "checkpoint191", **final_summary},
    ]


def write_manifest(metrics_rows, selected_count):
    baseline = metrics_rows[0]
    final = metrics_rows[1]
    lines = [
        "# Checkpoint191 checkpoint186 + MT190",
        "",
        "## Entradas",
        "",
        f"- Checkpoint186: `{CHECKPOINT186_DIR}`.",
        f"- MT190 validado: `{MT190_DIR}`.",
        "",
        "## Selecoes",
        "",
        "- PT mantido do checkpoint186.",
        f"- MT substituido pelo MT190 seguro em {selected_count} casos.",
        "- TL/L mantido do checkpoint186.",
        "",
        "## Métricas principais",
        "",
        "| Stage | Max Cobb SMAPE_article | PT | MT | TL/L | Agg3 | MAE3 | RMSE3 | <=5 | Falhas | Severos | Reg target >5 | Reg img >1 | Reg img >3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['stage']} | {row['max_cobb_smape_article']} | {row['pt_smape_article']} | "
            f"{row['mt_smape_article']} | {row['tll_smape_article']} | {row['agg3_smape_article']} | "
            f"{row['mae3']} | {row['rmse3']} | {row['within_5']} | {row['failures_gt5']} | "
            f"{row['severe_gt8']} | {row['regressions_target_gt5']} | {row['regressions_image_gt1']} | "
            f"{row['regressions_image_gt3']} |"
        )
    lines.extend([
        "",
        "## Leitura",
        "",
        f"- Falhas: {baseline['failures_gt5']} -> {final['failures_gt5']}.",
        f"- Severos: {baseline['severe_gt8']} -> {final['severe_gt8']}.",
        f"- MAE3: {baseline['mae3']} -> {final['mae3']}.",
        f"- <=5: {baseline['within_5']} -> {final['within_5']}.",
        "- Este script promove o candidato seguro validado no 190 para um checkpoint reprodutivel.",
    ])
    (OUTPUT_DIR / "checkpoint191_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    checkpoint186_target_rows = read_csv(CHECKPOINT186_DIR / "selector186_best_target_rows.csv")
    mt190_target_rows = read_csv(MT190_DIR / "mt190_best_safe_target_rows.csv")
    mt190_selected_rows = read_csv(MT190_DIR / "mt190_best_safe_selected_cases.csv")

    checkpoint191_target_rows, checkpoint191_image_rows = build_checkpoint_rows(mt190_target_rows)
    validate_against_mt190(checkpoint191_target_rows, checkpoint191_image_rows)
    metrics = metric_rows(checkpoint186_target_rows, checkpoint191_target_rows)

    write_csv(OUTPUT_DIR / "checkpoint191_target_rows.csv", checkpoint191_target_rows, list(checkpoint191_target_rows[0].keys()))
    write_csv(OUTPUT_DIR / "checkpoint191_image_rows.csv", checkpoint191_image_rows, list(checkpoint191_image_rows[0].keys()))
    write_csv(OUTPUT_DIR / "checkpoint191_selected_mt190_cases.csv", mt190_selected_rows, list(mt190_selected_rows[0].keys()))
    write_csv(OUTPUT_DIR / "checkpoint191_metrics_summary.csv", metrics, list(metrics[0].keys()))
    write_manifest(metrics, len(mt190_selected_rows))

    print("\n===== CHECKPOINT191 CHECKPOINT186 + MT190 =====")
    for row in metrics:
        print_required_metric_block(row["stage"], row)
    print(f"\nMT190 selecionados: {len(mt190_selected_rows)}")
    print(f"Manifesto: {OUTPUT_DIR / 'checkpoint191_manifest.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
