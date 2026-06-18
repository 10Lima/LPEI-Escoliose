import csv
import os
from collections import Counter, defaultdict
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
    "SPINAL_189_CHECKPOINT186_DIR",
    str(ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1"),
))
MT_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_189_MT_CANDIDATES_DIR",
    str(ACTIVE_DIR / "validar_selector_mt_diversidade_checkpoint114_global_top1200_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_189_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "auditar_mt188_regressoes_checkpoint186_v1"),
))

OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
TARGET = "MT"
CONFIG_ID = "188__pt_off__mt_r500_j16_s05__tll_off__sel_rank"
MAX_RANK = 500
MAX_JUMP = 16.0
MIN_SCORE = 0.50


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


def union_fieldnames(rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


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


def load_contexts():
    image_rows = read_csv(CHECKPOINT186_DIR / "selector186_best_image_rows.csv")
    target_rows = read_csv(CHECKPOINT186_DIR / "selector186_best_target_rows.csv")
    target_by_file = target_index(target_rows)
    contexts = {}
    for image in image_rows:
        file_id = image["file_id"]
        values = target_by_file[file_id]
        gt = [safe_float(values[target]["gt"]) for target in TARGETS]
        estimated = [safe_float(values[target]["estimated"]) for target in TARGETS]
        errors = [abs(estimated[idx] - gt[idx]) for idx in range(3)]
        contexts[file_id] = {
            "file_id": file_id,
            "raw_file_id": image.get("raw_file_id", values["PT"].get("raw_file_id", "")),
            "gt": gt,
            "base": estimated,
            "base_errors": errors,
            "base_mae3": float(np.mean(errors)),
            "base_severe": int(float(np.mean(errors)) > 8.0),
            "base_failure": int(float(np.mean(errors)) > 5.0),
            "mt_source186": values["MT"].get("source186", ""),
        }
    return contexts


def candidate_passes(item):
    return (
        int(item["rank"]) <= MAX_RANK
        and safe_float(item["jump"]) <= MAX_JUMP
        and safe_float(item["score"]) >= MIN_SCORE
    )


def stream_mt_candidates(contexts):
    path = MT_CANDIDATES_DIR / "selector125_oof_candidate_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV de candidatos MT nao encontrado: {path}")
    groups = defaultdict(list)
    wanted = set(contexts.keys())
    print(f"A ler candidatos MT: {path}", flush=True)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["file_id"]
            if file_id not in wanted:
                continue
            context = contexts[file_id]
            cobb = safe_float(row["cobb"])
            item = {
                "file_id": file_id,
                "rank": int(float(row["rank"])),
                "score": safe_float(row.get("predicted_good_score", row.get("selection_score", ""))),
                "selection_score": safe_float(row.get("selection_score", "")),
                "cobb": cobb,
                "label_abs_error": safe_float(row.get("label_abs_error", "")),
                "jump": abs(cobb - context["base"][1]),
                "signed_jump": cobb - context["base"][1],
                "window_name": row.get("window_name", ""),
                "norm_y_top": safe_float(row.get("norm_y_top", "")),
                "norm_y_bottom": safe_float(row.get("norm_y_bottom", "")),
                "norm_span": safe_float(row.get("norm_span", "")),
                "angle_top": safe_float(row.get("angle_top", "")),
                "angle_bottom": safe_float(row.get("angle_bottom", "")),
                "point_distance": safe_float(row.get("point_distance", "")),
                "smooth": safe_float(row.get("smooth", "")),
                "direction_up": safe_float(row.get("direction_up", "")),
            }
            if candidate_passes(item):
                groups[file_id].append(item)
    for file_id in list(groups.keys()):
        groups[file_id].sort(key=lambda item: (item["rank"], -safe_float(item["score"])))
    return groups


def selected_item(groups, file_id):
    items = groups.get(file_id, [])
    return items[0] if items else None


def arrays_for_method(contexts, groups):
    file_ids = sorted(contexts.keys())
    base_preds = []
    preds = []
    gts = []
    for file_id in file_ids:
        context = contexts[file_id]
        est = list(context["base"])
        item = selected_item(groups, file_id)
        if item is not None:
            est[1] = item["cobb"]
        base_preds.append(context["base"])
        preds.append(est)
        gts.append(context["gt"])
    return np.asarray(base_preds), np.asarray(preds), np.asarray(gts)


def classify_row(base_mae, new_mae, base_mt_error, new_mt_error):
    if new_mae - base_mae > 1.0:
        return "image_regression_gt1"
    if new_mae > 8.0 and base_mae <= 8.0:
        return "new_severe"
    if base_mae > 5.0 and new_mae <= 5.0:
        return "fixed_failure"
    if new_mt_error < base_mt_error - 1e-6:
        return "mt_improved"
    if new_mt_error > base_mt_error + 1e-6:
        return "mt_worsened"
    return "neutral"


def build_rows(contexts, groups):
    rows = []
    for file_id in sorted(contexts.keys()):
        context = contexts[file_id]
        item = selected_item(groups, file_id)
        if item is None:
            continue
        est = list(context["base"])
        est[1] = item["cobb"]
        new_errors = np.abs(np.asarray(est) - np.asarray(context["gt"]))
        new_mae = float(np.mean(new_errors))
        base_mae = context["base_mae3"]
        base_mt_error = context["base_errors"][1]
        new_mt_error = float(new_errors[1])
        rows.append({
            "file_id": file_id,
            "raw_file_id": context["raw_file_id"],
            "group": classify_row(base_mae, new_mae, base_mt_error, new_mt_error),
            "gt_pt": round_safe(context["gt"][0]),
            "gt_mt": round_safe(context["gt"][1]),
            "gt_tll": round_safe(context["gt"][2]),
            "base_pt": round_safe(context["base"][0]),
            "base_mt": round_safe(context["base"][1]),
            "base_tll": round_safe(context["base"][2]),
            "selected_mt": round_safe(item["cobb"]),
            "base_pt_abs_error": round_safe(context["base_errors"][0]),
            "base_mt_abs_error": round_safe(base_mt_error),
            "base_tll_abs_error": round_safe(context["base_errors"][2]),
            "new_mt_abs_error": round_safe(new_mt_error),
            "base_mae3": round_safe(base_mae),
            "new_mae3": round_safe(new_mae),
            "delta_mae3": round_safe(new_mae - base_mae),
            "delta_mt_abs_error": round_safe(new_mt_error - base_mt_error),
            "base_failure_gt5": context["base_failure"],
            "new_failure_gt5": int(new_mae > 5.0),
            "base_severe_gt8": context["base_severe"],
            "new_severe_gt8": int(new_mae > 8.0),
            "rank": item["rank"],
            "score": round_safe(item["score"]),
            "selection_score": round_safe(item["selection_score"]),
            "jump": round_safe(item["jump"]),
            "signed_jump": round_safe(item["signed_jump"]),
            "window_name": item["window_name"],
            "norm_y_top": round_safe(item["norm_y_top"]),
            "norm_y_bottom": round_safe(item["norm_y_bottom"]),
            "norm_span": round_safe(item["norm_span"]),
            "angle_top": round_safe(item["angle_top"]),
            "angle_bottom": round_safe(item["angle_bottom"]),
            "point_distance": round_safe(item["point_distance"]),
            "smooth": round_safe(item["smooth"]),
            "direction_up": round_safe(item["direction_up"]),
            "mt_source186": context["mt_source186"],
            "config_id": CONFIG_ID,
        })
    return rows


def summarize_groups(rows):
    counter = Counter(row["group"] for row in rows)
    summary = []
    for group, count in sorted(counter.items()):
        items = [row for row in rows if row["group"] == group]
        summary.append({
            "group": group,
            "cases": count,
            "mean_delta_mae3": round_safe(np.mean([safe_float(row["delta_mae3"]) for row in items])),
            "mean_delta_mt_abs_error": round_safe(np.mean([safe_float(row["delta_mt_abs_error"]) for row in items])),
            "median_rank": round_safe(np.median([safe_float(row["rank"]) for row in items])),
            "median_score": round_safe(np.median([safe_float(row["score"]) for row in items])),
            "median_jump": round_safe(np.median([safe_float(row["jump"]) for row in items])),
        })
    return summary


def heuristic_hints(rows):
    regressions = [row for row in rows if safe_float(row["delta_mae3"]) > 1.0]
    improved = [row for row in rows if safe_float(row["delta_mae3"]) < -1.0]
    hints = []
    for field in ["rank", "score", "jump", "signed_jump", "base_mt_abs_error", "norm_span", "norm_y_top", "norm_y_bottom"]:
        reg_values = np.array([safe_float(row[field]) for row in regressions], dtype=float)
        imp_values = np.array([safe_float(row[field]) for row in improved], dtype=float)
        reg_values = reg_values[np.isfinite(reg_values)]
        imp_values = imp_values[np.isfinite(imp_values)]
        if len(reg_values) == 0 or len(imp_values) == 0:
            continue
        hints.append({
            "field": field,
            "regression_median": round_safe(np.median(reg_values)),
            "improved_median": round_safe(np.median(imp_values)),
            "regression_p25": round_safe(np.percentile(reg_values, 25)),
            "regression_p75": round_safe(np.percentile(reg_values, 75)),
            "improved_p25": round_safe(np.percentile(imp_values, 25)),
            "improved_p75": round_safe(np.percentile(imp_values, 75)),
        })
    return hints


def write_summary(base_summary, mt_summary, group_summary, hints):
    lines = [
        "# Auditoria 189 MT188 regressões checkpoint186",
        "",
        "## Escopo",
        "",
        f"- Config auditada: `{CONFIG_ID}`.",
        f"- Regra: rank <= {MAX_RANK}, jump <= {MAX_JUMP}, score >= {MIN_SCORE}, selecao por rank.",
        "- PT e TL/L ficam iguais ao checkpoint186.",
        "- GT usado apenas para auditoria e métricas.",
        "",
        "## Métricas",
        "",
    ]
    for title, summary in [("checkpoint186", base_summary), ("mt188_audit", mt_summary)]:
        lines.append(f"### {title}")
        for key in [
            "max_cobb_smape_article", "pt_smape_article", "mt_smape_article",
            "tll_smape_article", "agg3_smape_article", "mae3", "rmse3",
            "within_5", "within_10", "within_15", "failures_gt5", "severe_gt8",
            "regressions_target_gt5", "regressions_image_gt1", "regressions_image_gt3",
        ]:
            lines.append(f"- {key}: {summary[key]}")
        lines.append("")
    lines.extend(["## Grupos", ""])
    for row in group_summary:
        lines.append(
            f"- {row['group']}: {row['cases']} casos | delta MAE3 medio {row['mean_delta_mae3']} | "
            f"rank mediano {row['median_rank']} | score mediano {row['median_score']} | jump mediano {row['median_jump']}"
        )
    lines.extend(["", "## Sinais para o 190", ""])
    for row in hints:
        lines.append(
            f"- {row['field']}: reg mediana {row['regression_median']} "
            f"vs melhorados mediana {row['improved_median']}"
        )
    lines.extend([
        "",
        "## Leitura",
        "",
        "- Se as regressões se separarem por rank, score, jump ou baseline MT error, criar 190 MT protegido.",
        "- Se não houver separação clara, não continuar com este MT selector simples.",
    ])
    (OUTPUT_DIR / "audit189_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    contexts = load_contexts()
    groups = stream_mt_candidates(contexts)
    base_preds, preds, gts = arrays_for_method(contexts, groups)
    base_summary = metric_summary_from_arrays(base_preds, gts)
    mt_summary = metric_summary_from_arrays(preds, gts, baseline_preds=base_preds)
    rows = build_rows(contexts, groups)
    group_summary = summarize_groups(rows)
    hints = heuristic_hints(rows)

    write_csv(OUTPUT_DIR / "audit189_selected_mt_rows.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "audit189_regressions_image_gt1.csv", [row for row in rows if safe_float(row["delta_mae3"]) > 1.0], union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "audit189_new_severes.csv", [row for row in rows if safe_float(row["new_mae3"]) > 8.0 and safe_float(row["base_mae3"]) <= 8.0], union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "audit189_fixed_failures.csv", [row for row in rows if safe_float(row["base_mae3"]) > 5.0 and safe_float(row["new_mae3"]) <= 5.0], union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "audit189_group_summary.csv", group_summary, union_fieldnames(group_summary))
    write_csv(OUTPUT_DIR / "audit189_heuristic_hints.csv", hints, union_fieldnames(hints) if hints else ["field"])
    write_summary(base_summary, mt_summary, group_summary, hints)

    print("\n===== AUDITORIA189 MT188 REGRESSOES =====")
    print_required_metric_block("checkpoint186", base_summary)
    print_required_metric_block("mt188 auditado", mt_summary)
    print("\nGrupos:")
    for row in group_summary:
        print(f"  {row['group']}: {row['cases']}")
    print(f"\nResumo: {OUTPUT_DIR / 'audit189_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
