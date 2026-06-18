import csv
import os
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import TARGETS, metric_summary_from_arrays, round_safe, safe_float


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"
COMPARISON_DIR = (
    PROCESSED_DIR
    / "cobb_results"
    / "comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1"
)

CHECKPOINT205_TARGET_ROWS = ACTIVE_DIR / "checkpoint205_checkpoint201_sf204_v1" / "checkpoint205_target_rows.csv"
TRAIN2000_248_TARGET_ROWS = COMPARISON_DIR / "248_train2000_target_rows.csv"
TRAINFULL_248_TARGET_ROWS = COMPARISON_DIR / "248_trainfull_target_rows.csv"
OVERWRITE = os.environ.get("SPINAL_249_OVERWRITE", "0").strip() == "1"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if path.exists() and not OVERWRITE:
        raise FileExistsError(f"Output ja existe: {path}. Usa SPINAL_249_OVERWRITE=1 para substituir.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_by_raw_id(target_rows):
    grouped = {}
    for row in target_rows:
        raw_id = row.get("raw_file_id", "")
        target = row.get("target", "")
        if target in TARGETS:
            grouped.setdefault(raw_id, {})[target] = row
    return {
        raw_id: values
        for raw_id, values in grouped.items()
        if all(target in values for target in TARGETS)
    }


def arrays_for_raw_ids(grouped, raw_ids):
    preds = []
    gts = []
    kept = []
    for raw_id in sorted(raw_ids):
        values = grouped.get(raw_id)
        if values is None:
            continue
        preds.append([safe_float(values[target]["estimated"]) for target in TARGETS])
        gts.append([safe_float(values[target]["gt"]) for target in TARGETS])
        kept.append(raw_id)
    return kept, np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)


def summarize_group(label, source, grouped, raw_ids, baseline_preds=None):
    kept, preds, gts = arrays_for_raw_ids(grouped, raw_ids)
    if not kept:
        raise RuntimeError(f"Grupo vazio: {source} {label}")
    summary = metric_summary_from_arrays(preds, gts, baseline_preds=baseline_preds)
    return {"source": source, "group": label, "raw_id_count": len(kept), **summary}, kept, preds, gts


def source_rows_for_group(source, grouped, label, raw_ids):
    rows = []
    for raw_id in sorted(raw_ids):
        values = grouped.get(raw_id)
        if values is None:
            continue
        errors = [abs(safe_float(values[target]["estimated"]) - safe_float(values[target]["gt"])) for target in TARGETS]
        rows.append({
            "source": source,
            "group": label,
            "raw_file_id": raw_id,
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(float(np.mean(errors))),
            "within_5": int(float(np.mean(errors)) <= 5.0),
            "pt_source": values["PT"].get("source248", values["PT"].get("source205", "")),
            "mt_source": values["MT"].get("source248", values["MT"].get("source205", "")),
            "tll_source": values["TL_L"].get("source248", values["TL_L"].get("source205", "")),
        })
    return rows


def comparison_rows(metrics_rows):
    by_key = {(row["source"], row["group"]): row for row in metrics_rows}
    rows = []
    for group in ["checkpoint205_overlap_1000", "remaining_2192", "all_3192"]:
        train2000 = by_key.get(("train2000_248", group))
        trainfull = by_key.get(("trainfull_248", group))
        if not train2000 or not trainfull:
            continue
        for metric in ["within_5", "mae3", "rmse3", "failures_gt5", "severe_gt8"]:
            rows.append({
                "group": group,
                "metric": metric,
                "train2000_248": train2000[metric],
                "trainfull_248": trainfull[metric],
                "delta_trainfull_minus_train2000": round_safe(safe_float(trainfull[metric]) - safe_float(train2000[metric])),
            })
    return rows


def write_summary(metrics_rows, overlap_count, missing_from_248):
    by_key = {(row["source"], row["group"]): row for row in metrics_rows}
    lines = [
        "# Auditoria 249 - 248 nos 1000 do checkpoint205 vs restantes",
        "",
        "## Cruzamento",
        "",
        f"- Raw IDs no checkpoint205 historico: 1000.",
        f"- Encontrados no 248 por raw_file_id: {overlap_count}.",
        f"- Ausentes no 248: {missing_from_248}.",
        "",
        "## Metricas principais",
        "",
    ]
    for source in ["checkpoint205_historico", "train2000_248", "trainfull_248"]:
        for group in ["checkpoint205_overlap_1000", "remaining_2192", "all_3192"]:
            row = by_key.get((source, group))
            if row is None:
                continue
            lines.extend([
                f"### {source} / {group}",
                f"- casos: {row['raw_id_count']}",
                f"- <=5: {row['within_5']}%",
                f"- MAE3: {row['mae3']}",
                f"- RMSE3: {row['rmse3']}",
                f"- falhas >5: {row['failures_gt5']}",
                f"- severos >8: {row['severe_gt8']}",
                "",
            ])
    lines.extend([
        "## Leitura",
        "",
        "- Se o 248 nos 1000 do checkpoint205 ficar longe do checkpoint205 historico, o problema principal esta nos candidatos/gates regenerados.",
        "- Se o 248 nos 1000 ficar perto e cair nos restantes 2192, o problema principal e generalizacao para val_all_3192.",
    ])
    (COMPARISON_DIR / "249_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    checkpoint205 = rows_by_raw_id(read_csv(CHECKPOINT205_TARGET_ROWS))
    train2000 = rows_by_raw_id(read_csv(TRAIN2000_248_TARGET_ROWS))
    trainfull = rows_by_raw_id(read_csv(TRAINFULL_248_TARGET_ROWS))

    checkpoint_raw_ids = set(checkpoint205)
    all_raw_ids = set(train2000) & set(trainfull)
    overlap_ids = checkpoint_raw_ids & all_raw_ids
    remaining_ids = all_raw_ids - checkpoint_raw_ids
    missing = checkpoint_raw_ids - all_raw_ids

    metrics_rows = []
    image_rows = []

    row, kept, checkpoint_preds, checkpoint_gts = summarize_group(
        "checkpoint205_overlap_1000", "checkpoint205_historico", checkpoint205, overlap_ids
    )
    metrics_rows.append(row)
    image_rows.extend(source_rows_for_group("checkpoint205_historico", checkpoint205, "checkpoint205_overlap_1000", kept))

    for source, grouped in [("train2000_248", train2000), ("trainfull_248", trainfull)]:
        for group, ids in [
            ("checkpoint205_overlap_1000", overlap_ids),
            ("remaining_2192", remaining_ids),
            ("all_3192", all_raw_ids),
        ]:
            baseline = checkpoint_preds if group == "checkpoint205_overlap_1000" and checkpoint_preds.shape[0] == len(overlap_ids) else None
            row, kept, _preds, _gts = summarize_group(group, source, grouped, ids, baseline_preds=baseline)
            metrics_rows.append(row)
            image_rows.extend(source_rows_for_group(source, grouped, group, kept))

    compare = comparison_rows(metrics_rows)
    write_csv(COMPARISON_DIR / "249_group_metrics_summary.csv", metrics_rows)
    write_csv(COMPARISON_DIR / "249_train2000_vs_trainfull_by_group.csv", compare)
    write_csv(COMPARISON_DIR / "249_image_rows_by_group.csv", image_rows)
    write_summary(metrics_rows, len(overlap_ids), len(missing))

    for row in metrics_rows:
        print(
            f"{row['source']} {row['group']}: casos={row['raw_id_count']} "
            f"<=5={row['within_5']} MAE3={row['mae3']} falhas={row['failures_gt5']}"
        )
    print(f"Resumo: {COMPARISON_DIR / '249_summary.md'}")


if __name__ == "__main__":
    main()
