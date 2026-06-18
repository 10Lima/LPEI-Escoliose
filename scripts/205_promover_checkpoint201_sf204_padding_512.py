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

CHECKPOINT201_DIR = Path(os.environ.get(
    "SPINAL_205_CHECKPOINT201_DIR",
    str(ACTIVE_DIR / "checkpoint201_checkpoint198_pt200_v1"),
))
SF204_DIR = Path(os.environ.get(
    "SPINAL_205_SF204_DIR",
    str(ACTIVE_DIR / "validar_severe_first_pt_tll_checkpoint201_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_205_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "checkpoint205_checkpoint201_sf204_v1"),
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
    for file_id in sorted(by_file):
        preds.append([safe_float(by_file[file_id][target]["estimated"]) for target in TARGETS])
        gts.append([safe_float(by_file[file_id][target]["gt"]) for target in TARGETS])
    return np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)


def build_checkpoint_rows(sf204_target_rows):
    target_rows = []
    for row in sf204_target_rows:
        source204 = row["source204"]
        target_rows.append({
            "file_id": row["file_id"],
            "raw_file_id": row.get("raw_file_id", ""),
            "target": row["target"],
            "gt": row["gt"],
            "estimated": row["estimated"],
            "abs_error": row["abs_error"],
            "source205": source204,
            "source204": source204,
        })

    by_file = target_index(target_rows)
    image_rows = []
    for file_id in sorted(by_file):
        values = by_file[file_id]
        errors = [safe_float(values[target]["abs_error"]) for target in TARGETS]
        changed = [target for target in TARGETS if str(values[target]["source205"]).startswith("sf204_")]
        image_rows.append({
            "file_id": file_id,
            "raw_file_id": values["PT"].get("raw_file_id", ""),
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(float(np.mean(errors))),
            "within_5": int(float(np.mean(errors)) <= 5.0),
            "pt_source205": values["PT"]["source205"],
            "mt_source205": values["MT"]["source205"],
            "tll_source205": values["TL_L"]["source205"],
            "changed_targets205": "+".join(changed),
        })
    return target_rows, image_rows


def validate_against_sf204(checkpoint_target_rows, checkpoint_image_rows):
    ref_target_rows = read_csv(SF204_DIR / "sf204_best_safe_target_rows.csv")
    ref_image_rows = read_csv(SF204_DIR / "sf204_best_safe_image_rows.csv")
    own_targets = target_index(checkpoint_target_rows)
    ref_targets = target_index(ref_target_rows)
    if set(own_targets) != set(ref_targets):
        raise RuntimeError("Ficheiros alvo divergem entre checkpoint205 e sf204.")
    for file_id in own_targets:
        for target in TARGETS:
            own = own_targets[file_id][target]
            ref = ref_targets[file_id][target]
            for key in ["gt", "estimated", "abs_error"]:
                if abs(safe_float(own[key]) - safe_float(ref[key])) > 1e-4:
                    raise RuntimeError(f"Divergencia em {file_id} {target} {key}")

    own_images = {row["file_id"]: row for row in checkpoint_image_rows}
    ref_images = {row["file_id"]: row for row in ref_image_rows}
    if set(own_images) != set(ref_images):
        raise RuntimeError("Ficheiros imagem divergem entre checkpoint205 e sf204.")
    for file_id, own in own_images.items():
        ref = ref_images[file_id]
        for key in ["pt_abs_error", "mt_abs_error", "tll_abs_error", "mae3"]:
            if abs(safe_float(own[key]) - safe_float(ref[key])) > 1e-4:
                raise RuntimeError(f"Divergencia imagem {file_id} {key}")


def validate_no_unexpected_changes(checkpoint201_target_rows, checkpoint205_target_rows):
    base = target_index(checkpoint201_target_rows)
    final = target_index(checkpoint205_target_rows)
    if set(base) != set(final):
        raise RuntimeError("Ficheiros divergem entre checkpoint201 e checkpoint205.")
    changed = []
    for file_id in sorted(base):
        for target in TARGETS:
            base_row = base[file_id][target]
            final_row = final[file_id][target]
            if abs(safe_float(base_row["gt"]) - safe_float(final_row["gt"])) > 1e-4:
                raise RuntimeError(f"GT divergente em {file_id} {target}")
            delta_est = abs(safe_float(base_row["estimated"]) - safe_float(final_row["estimated"]))
            if delta_est <= 1e-4:
                continue
            if target not in {"PT", "TL_L"}:
                raise RuntimeError(f"Alteracao inesperada fora de PT/TL_L: {file_id} {target}")
            changed.append((file_id, target))
    return changed


def metric_rows(checkpoint201_target_rows, checkpoint205_target_rows):
    base_preds, gts = arrays_from_target_rows(checkpoint201_target_rows)
    preds, gts205 = arrays_from_target_rows(checkpoint205_target_rows)
    if not np.allclose(gts, gts205):
        raise RuntimeError("GT divergente entre checkpoint201 e checkpoint205.")
    return [
        {"stage": "checkpoint201", **metric_summary_from_arrays(base_preds, gts)},
        {"stage": "checkpoint205", **metric_summary_from_arrays(preds, gts, baseline_preds=base_preds)},
    ]


def write_manifest(metrics_rows, selected_count):
    baseline = metrics_rows[0]
    final = metrics_rows[1]
    lines = [
        "# Checkpoint205 checkpoint201 + SF204",
        "",
        "## Entradas",
        "",
        f"- Checkpoint201: `{CHECKPOINT201_DIR}`.",
        f"- SF204 validado: `{SF204_DIR}`.",
        "",
        "## Selecoes",
        "",
        f"- Substituicoes severe-first seguras em {selected_count} targets.",
        "- MT mantido do checkpoint201.",
        "",
        "## Metricas principais",
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
        "- Este script promove o candidato seguro validado no 204 para um checkpoint reprodutivel.",
    ])
    (OUTPUT_DIR / "checkpoint205_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    checkpoint201_target_rows = read_csv(CHECKPOINT201_DIR / "checkpoint201_target_rows.csv")
    sf204_target_rows = read_csv(SF204_DIR / "sf204_best_safe_target_rows.csv")
    sf204_selected_rows = read_csv(SF204_DIR / "sf204_best_safe_selected_cases.csv")

    checkpoint205_target_rows, checkpoint205_image_rows = build_checkpoint_rows(sf204_target_rows)
    validate_against_sf204(checkpoint205_target_rows, checkpoint205_image_rows)
    changed_targets = validate_no_unexpected_changes(checkpoint201_target_rows, checkpoint205_target_rows)
    if len(changed_targets) != len(sf204_selected_rows):
        raise RuntimeError(
            f"Selecionados SF204 ({len(sf204_selected_rows)}) divergem dos alterados ({len(changed_targets)})."
        )
    metrics = metric_rows(checkpoint201_target_rows, checkpoint205_target_rows)

    write_csv(OUTPUT_DIR / "checkpoint205_target_rows.csv", checkpoint205_target_rows, list(checkpoint205_target_rows[0].keys()))
    write_csv(OUTPUT_DIR / "checkpoint205_image_rows.csv", checkpoint205_image_rows, list(checkpoint205_image_rows[0].keys()))
    write_csv(OUTPUT_DIR / "checkpoint205_selected_sf204_cases.csv", sf204_selected_rows, list(sf204_selected_rows[0].keys()))
    write_csv(OUTPUT_DIR / "checkpoint205_metrics_summary.csv", metrics, list(metrics[0].keys()))
    write_manifest(metrics, len(sf204_selected_rows))

    print("\n===== CHECKPOINT205 CHECKPOINT201 + SF204 =====")
    for row in metrics:
        print_required_metric_block(row["stage"], row)
    print(f"\nSF204 selecionados: {len(sf204_selected_rows)}")
    print(f"Manifesto: {OUTPUT_DIR / 'checkpoint205_manifest.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
