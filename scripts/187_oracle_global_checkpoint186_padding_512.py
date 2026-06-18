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
    safe_int,
    smape_target_article,
    smape_target_old,
)


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT186_DIR = Path(os.environ.get(
    "SPINAL_187_CHECKPOINT186_DIR",
    str(ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1"),
))
PT_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_187_PT_CANDIDATES_DIR",
    str(ACTIVE_DIR / "gate_pt_hard_negatives_checkpoint114_v1"),
))
MT_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_187_MT_CANDIDATES_DIR",
    str(ACTIVE_DIR / "validar_selector_mt_diversidade_checkpoint114_global_top1200_v1"),
))
TLL_OLD_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_187_TLL_OLD_CANDIDATES_DIR",
    str(ACTIVE_DIR / "gate_two_stage_hard_negatives_tll_v1"),
))
TLL159_DIR = Path(os.environ.get(
    "SPINAL_187_TLL159_DIR",
    str(ACTIVE_DIR / "gerar_candidatos_tll_checkpoint156_relevantes_v1"),
))
TLL175_DIR = Path(os.environ.get(
    "SPINAL_187_TLL175_DIR",
    str(ACTIVE_DIR / "gerar_candidatos_tll_remaining_selector173_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_187_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "oracle_global_checkpoint186_v1"),
))
MAX_FILES = safe_int(os.environ.get("SPINAL_187_MAX_FILES", "0"))

OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
SHORTLIST_RANK_MAX = 25


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
        for key in row.keys():
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


def load_checkpoint186():
    image_rows = read_csv(CHECKPOINT186_DIR / "selector186_best_image_rows.csv")
    target_rows = read_csv(CHECKPOINT186_DIR / "selector186_best_target_rows.csv")
    target_by_file = target_index(target_rows)
    file_ids = sorted(row["file_id"] for row in image_rows)
    if MAX_FILES > 0:
        file_ids = file_ids[:MAX_FILES]
    contexts = {}
    for file_id in file_ids:
        values = target_by_file[file_id]
        gt = [safe_float(values[target]["gt"]) for target in TARGETS]
        estimated = [safe_float(values[target]["estimated"]) for target in TARGETS]
        contexts[file_id] = {
            "file_id": file_id,
            "raw_file_id": values["PT"].get("raw_file_id", ""),
            "gt": gt,
            "checkpoint186": estimated,
            "checkpoint186_errors": [abs(estimated[idx] - gt[idx]) for idx in range(3)],
            "sources186": {target: values[target].get("source186", "") for target in TARGETS},
        }
    return contexts


def candidate_specs():
    return [
        {
            "target": "PT",
            "path": PT_CANDIDATES_DIR / "gate123_oof_candidate_predictions.csv",
            "source": "old_PT_gate123",
            "rank_col": "rank",
            "score_cols": ["gate123_score", "predicted_good_score", "selection_score"],
            "method_col": "window_name",
        },
        {
            "target": "MT",
            "path": MT_CANDIDATES_DIR / "selector125_oof_candidate_predictions.csv",
            "source": "old_MT_selector125",
            "rank_col": "rank",
            "score_cols": ["predicted_good_score", "selection_score"],
            "method_col": "window_name",
        },
        {
            "target": "TL_L",
            "path": TLL_OLD_CANDIDATES_DIR / "gate114_oof_candidate_predictions.csv",
            "source": "old_TLL_gate114",
            "rank_col": "rank",
            "score_cols": ["gate114_score", "predicted_good_score", "selection_score"],
            "method_col": "",
        },
        {
            "target": "TL_L",
            "path": TLL159_DIR / "tll159_candidate_rows.csv",
            "source": "tll159",
            "rank_col": "rank159",
            "score_cols": ["selection_score"],
            "method_col": "method159",
        },
        {
            "target": "TL_L",
            "path": TLL175_DIR / "tll175_candidate_rows.csv",
            "source": "tll175",
            "rank_col": "rank175",
            "score_cols": ["selection_score"],
            "method_col": "method175",
        },
    ]


def first_score(row, cols):
    for col in cols:
        value = safe_float(row.get(col, ""))
        if np.isfinite(value):
            return value
    return np.nan


def better_candidate(current, item):
    if current is None:
        return True
    return (
        safe_float(item["abs_error"]),
        safe_int(item["rank"]),
        -safe_float(item["score"]),
    ) < (
        safe_float(current["abs_error"]),
        safe_int(current["rank"]),
        -safe_float(current["score"]),
    )


def stream_candidate_oracles(contexts):
    wanted_files = set(contexts.keys())
    total_best = {}
    shortlist_best = {}
    counts = defaultdict(int)
    shortlist_counts = defaultdict(int)
    good_counts = defaultdict(int)
    good_shortlist_counts = defaultdict(int)

    for spec in candidate_specs():
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"CSV de candidatos nao encontrado: {path}")
        print(f"A ler candidatos {spec['source']}: {path}", flush=True)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row.get("file_id", "")
                if file_id not in wanted_files:
                    continue
                target = spec["target"]
                key = (file_id, target)
                rank = safe_int(row.get(spec["rank_col"], ""))
                abs_error = safe_float(row.get("label_abs_error", ""))
                cobb = safe_float(row.get("cobb", ""))
                if not np.isfinite(abs_error) or not np.isfinite(cobb):
                    continue
                item = {
                    "file_id": file_id,
                    "target": target,
                    "source": spec["source"],
                    "method": row.get(spec["method_col"], "") if spec["method_col"] else spec["source"],
                    "rank": rank,
                    "cobb": cobb,
                    "abs_error": abs_error,
                    "score": first_score(row, spec["score_cols"]),
                }
                counts[key] += 1
                good_counts[key] += int(abs_error <= 5.0)
                if better_candidate(total_best.get(key), item):
                    total_best[key] = item
                if rank <= SHORTLIST_RANK_MAX:
                    shortlist_counts[key] += 1
                    good_shortlist_counts[key] += int(abs_error <= 5.0)
                    if better_candidate(shortlist_best.get(key), item):
                        shortlist_best[key] = item
    return total_best, shortlist_best, counts, shortlist_counts, good_counts, good_shortlist_counts


def fallback_candidate(context, target):
    idx = TARGETS.index(target)
    return {
        "source": "checkpoint186",
        "method": context["sources186"][target],
        "rank": "",
        "cobb": context["checkpoint186"][idx],
        "abs_error": context["checkpoint186_errors"][idx],
        "score": "",
    }


def choose_oracle(context, target, candidate):
    base = fallback_candidate(context, target)
    if candidate is None:
        return base, "checkpoint186_fallback"
    if safe_float(candidate["abs_error"]) < safe_float(base["abs_error"]):
        return candidate, "candidate"
    return base, "checkpoint186_fallback"


def build_method_arrays(contexts, total_best, shortlist_best):
    method_preds = {
        "checkpoint186": [],
        "oracle_total": [],
        "oracle_shortlist": [],
        "best_exported": [],
    }
    gts = []
    detail = {}
    for file_id in sorted(contexts.keys()):
        context = contexts[file_id]
        gts.append(context["gt"])
        method_values = {method: [] for method in method_preds}
        for target in TARGETS:
            key = (file_id, target)
            base = fallback_candidate(context, target)
            total_item, total_choice = choose_oracle(context, target, total_best.get(key))
            short_item, short_choice = choose_oracle(context, target, shortlist_best.get(key))
            best_exported = total_best.get(key) or base
            method_values["checkpoint186"].append(base["cobb"])
            method_values["oracle_total"].append(total_item["cobb"])
            method_values["oracle_shortlist"].append(short_item["cobb"])
            method_values["best_exported"].append(best_exported["cobb"])
            detail[key] = {
                "base": base,
                "oracle_total": total_item,
                "oracle_total_choice": total_choice,
                "oracle_shortlist": short_item,
                "oracle_shortlist_choice": short_choice,
                "best_exported": best_exported,
            }
        for method in method_preds:
            method_preds[method].append(method_values[method])
    return {
        method: np.asarray(values, dtype=float)
        for method, values in method_preds.items()
    }, np.asarray(gts, dtype=float), detail


def build_summaries(method_preds, gts):
    baseline = method_preds["checkpoint186"]
    names = {
        "checkpoint186": "checkpoint186",
        "oracle_total": "oracle_total",
        "oracle_shortlist": "oracle_shortlist",
        "best_exported": "best_exported_or_checkpoint_fallback",
    }
    rows = []
    for method, preds in method_preds.items():
        summary = metric_summary_from_arrays(preds, gts, baseline_preds=None if method == "checkpoint186" else baseline)
        summary = {"method": names[method], **summary}
        rows.append(summary)
    return rows


def build_diagnostic_rows(contexts, detail, counts, shortlist_counts, good_counts, good_shortlist_counts):
    rows = []
    for file_id in sorted(contexts.keys()):
        context = contexts[file_id]
        for idx, target in enumerate(TARGETS):
            key = (file_id, target)
            base = detail[key]["base"]
            total = detail[key]["oracle_total"]
            short = detail[key]["oracle_shortlist"]
            exported = detail[key]["best_exported"]
            shortlist_has_good = good_shortlist_counts.get(key, 0) > 0
            total_has_good = good_counts.get(key, 0) > 0
            chosen_shortlist_best = (
                shortlist_has_good
                and abs(safe_float(short["cobb"]) - safe_float(base["cobb"])) <= 1e-3
            )
            rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "gt": round_safe(context["gt"][idx]),
                "checkpoint186_estimated": round_safe(base["cobb"]),
                "checkpoint186_abs_error": round_safe(base["abs_error"]),
                "checkpoint186_source": base["method"],
                "candidate_count_total": counts.get(key, 0),
                "candidate_count_shortlist_rank_le25": shortlist_counts.get(key, 0),
                "good_candidate_total_le5": int(total_has_good),
                "good_candidate_shortlist_le5": int(shortlist_has_good),
                "good_candidate_count_total_le5": good_counts.get(key, 0),
                "good_candidate_count_shortlist_le5": good_shortlist_counts.get(key, 0),
                "oracle_total_estimated": round_safe(total["cobb"]),
                "oracle_total_abs_error": round_safe(total["abs_error"]),
                "oracle_total_source": total["source"],
                "oracle_total_method": total["method"],
                "oracle_total_rank": total["rank"],
                "oracle_total_score": round_safe(total["score"]),
                "oracle_total_uses_candidate": int(detail[key]["oracle_total_choice"] == "candidate"),
                "oracle_shortlist_estimated": round_safe(short["cobb"]),
                "oracle_shortlist_abs_error": round_safe(short["abs_error"]),
                "oracle_shortlist_source": short["source"],
                "oracle_shortlist_method": short["method"],
                "oracle_shortlist_rank": short["rank"],
                "oracle_shortlist_score": round_safe(short["score"]),
                "oracle_shortlist_uses_candidate": int(detail[key]["oracle_shortlist_choice"] == "candidate"),
                "best_exported_estimated": round_safe(exported["cobb"]),
                "best_exported_abs_error": round_safe(exported["abs_error"]),
                "best_exported_source": exported["source"],
                "best_exported_method": exported["method"],
                "best_exported_rank": exported["rank"],
                "best_exported_score": round_safe(exported["score"]),
                "candidate_enters_shortlist_not_chosen": int(
                    safe_float(base["abs_error"]) > 5.0 and shortlist_has_good and not chosen_shortlist_best
                ),
                "smape_target_article_checkpoint186": round_safe(
                    smape_target_article(base["cobb"], context["gt"][idx])
                ),
                "smape_target_article_oracle_total": round_safe(
                    smape_target_article(total["cobb"], context["gt"][idx])
                ),
                "smape_target_article_oracle_shortlist": round_safe(
                    smape_target_article(short["cobb"], context["gt"][idx])
                ),
                "smape_target_old_checkpoint186": round_safe(
                    smape_target_old(base["cobb"], context["gt"][idx])
                ),
            })
    return rows


def build_image_rows(contexts, method_preds, gts):
    rows = []
    file_ids = sorted(contexts.keys())
    base_errors = np.abs(method_preds["checkpoint186"] - gts)
    for image_idx, file_id in enumerate(file_ids):
        context = contexts[file_id]
        row = {
            "file_id": file_id,
            "raw_file_id": context["raw_file_id"],
            "gt_pt": round_safe(gts[image_idx, 0]),
            "gt_mt": round_safe(gts[image_idx, 1]),
            "gt_tll": round_safe(gts[image_idx, 2]),
        }
        for method, preds in method_preds.items():
            errors = np.abs(preds[image_idx] - gts[image_idx])
            row[f"{method}_pt"] = round_safe(preds[image_idx, 0])
            row[f"{method}_mt"] = round_safe(preds[image_idx, 1])
            row[f"{method}_tll"] = round_safe(preds[image_idx, 2])
            row[f"{method}_mae3"] = round_safe(np.mean(errors))
            row[f"{method}_max_abs_error"] = round_safe(np.max(errors))
            if method != "checkpoint186":
                row[f"{method}_delta_mae3_vs_186"] = round_safe(np.mean(errors) - np.mean(base_errors[image_idx]))
        rows.append(row)
    return rows


def recovery_counts(method_preds, gts):
    base_mae = np.mean(np.abs(method_preds["checkpoint186"] - gts), axis=1)
    base_fail = base_mae > 5.0
    base_severe = base_mae > 8.0
    rows = []
    for method in ["oracle_total", "oracle_shortlist", "best_exported"]:
        mae = np.mean(np.abs(method_preds[method] - gts), axis=1)
        rows.append({
            "method": method,
            "baseline_failures_gt5": int(np.sum(base_fail)),
            "baseline_severes_gt8": int(np.sum(base_severe)),
            "failures_correctable_to_le5": int(np.sum(base_fail & (mae <= 5.0))),
            "severes_correctable_to_le8": int(np.sum(base_severe & (mae <= 8.0))),
        })
    return rows


def diagnostic_counts(target_rows):
    failing = [row for row in target_rows if safe_float(row["checkpoint186_abs_error"]) > 5.0]
    by_target = Counter(row["target"] for row in failing)
    selector_gate = [
        row for row in failing
        if safe_int(row["good_candidate_shortlist_le5"]) == 1
        and safe_int(row["candidate_enters_shortlist_not_chosen"]) == 1
    ]
    shortlist_gap = [
        row for row in failing
        if safe_int(row["good_candidate_total_le5"]) == 1
        and safe_int(row["good_candidate_shortlist_le5"]) == 0
    ]
    generation_gap = [
        row for row in failing
        if safe_int(row["good_candidate_total_le5"]) == 0
    ]
    rows = [
        {"diagnostic": "target_failures_gt5", "count": len(failing)},
        {"diagnostic": "target_failures_pt_gt5", "count": by_target["PT"]},
        {"diagnostic": "target_failures_mt_gt5", "count": by_target["MT"]},
        {"diagnostic": "target_failures_tll_gt5", "count": by_target["TL_L"]},
        {"diagnostic": "selector_gate_or_choice_gap", "count": len(selector_gate)},
        {"diagnostic": "shortlist_gap_good_total_not_shortlist", "count": len(shortlist_gap)},
        {"diagnostic": "generation_or_geometry_gap_no_good_total", "count": len(generation_gap)},
    ]
    return rows


def conclusion_lines(summary_by_method, recovery_rows, diagnostic_rows):
    checkpoint = summary_by_method["checkpoint186"]
    oracle_total = summary_by_method["oracle_total"]
    oracle_short = summary_by_method["oracle_shortlist"]
    total_gain_fail = checkpoint["failures_gt5"] - oracle_total["failures_gt5"]
    short_gain_fail = checkpoint["failures_gt5"] - oracle_short["failures_gt5"]
    diag = {row["diagnostic"]: row["count"] for row in diagnostic_rows}
    lines = []
    if total_gain_fail >= 10:
        lines.append("- O oracle_total ainda mostra margem material face ao 186.")
    else:
        lines.append("- O oracle_total esta relativamente perto do 186; a margem estrutural pode estar limitada.")
    if oracle_short["failures_gt5"] > oracle_total["failures_gt5"] + 5:
        lines.append("- A shortlist perde uma parte relevante da margem; melhorar bandas/shortlist deve vir antes de novo selector.")
    elif diag["selector_gate_or_choice_gap"] > diag["shortlist_gap_good_total_not_shortlist"]:
        lines.append("- Ha sinal de problema no selector/gate: bons candidatos ja entram na shortlist mas nao ficam escolhidos.")
    else:
        lines.append("- A diferenca entre shortlist e total deve guiar se o proximo passo e shortlist ou selector.")
    if diag["target_failures_tll_gt5"] >= max(diag["target_failures_pt_gt5"], diag["target_failures_mt_gt5"]):
        lines.append("- TL/L continua candidato natural para ataque adaptativo, mas so se a margem oracle TL/L for real.")
    else:
        lines.append("- PT/MT parecem competir com TL/L como gargalo; a decisao deve seguir a decomposicao por target.")
    lines.append(
        f"- Das falhas >5 do 186, oracle_total corrige {total_gain_fail} e oracle_shortlist corrige {short_gain_fail}."
    )
    return lines


def write_summary(method_rows, recovery_rows, diagnostic_rows):
    summary_by_method = {row["method"]: row for row in method_rows}
    lines = [
        "# Oracle global checkpoint186",
        "",
        "## Escopo",
        "",
        f"- Checkpoint avaliado: `{CHECKPOINT186_DIR}`.",
        f"- Output: `{OUTPUT_DIR}`.",
        "- GT e usado apenas para diagnostico oracle; nao e uma regra de inferencia.",
        "- SMAPE principal usa a escala do artigo: fator 100 sobre soma global dos erros.",
        "",
        "## Metricas principais",
        "",
        "| Metodo | Max Cobb SMAPE_article | PT | MT | TL/L | Agg3 | MAE3 | RMSE3 | <=5 | Falhas | Severos | Reg target >5 | Reg img >1 | Reg img >3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['max_cobb_smape_article']} | {row['pt_smape_article']} | "
            f"{row['mt_smape_article']} | {row['tll_smape_article']} | {row['agg3_smape_article']} | "
            f"{row['mae3']} | {row['rmse3']} | {row['within_5']} | {row['failures_gt5']} | "
            f"{row['severe_gt8']} | {row['regressions_target_gt5']} | {row['regressions_image_gt1']} | "
            f"{row['regressions_image_gt3']} |"
        )
    lines.extend(["", "## Recuperacao oracle", ""])
    for row in recovery_rows:
        lines.append(
            f"- {row['method']}: corrige {row['failures_correctable_to_le5']} / "
            f"{row['baseline_failures_gt5']} falhas >5 e {row['severes_correctable_to_le8']} / "
            f"{row['baseline_severes_gt8']} severos."
        )
    lines.extend(["", "## Diagnostico", ""])
    for row in diagnostic_rows:
        lines.append(f"- {row['diagnostic']}: {row['count']}")
    lines.extend(["", "## Conclusao", ""])
    lines.extend(conclusion_lines(summary_by_method, recovery_rows, diagnostic_rows))
    (OUTPUT_DIR / "oracle187_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    contexts = load_checkpoint186()
    total_best, shortlist_best, counts, shortlist_counts, good_counts, good_shortlist_counts = stream_candidate_oracles(contexts)
    method_preds, gts, detail = build_method_arrays(contexts, total_best, shortlist_best)
    method_rows = build_summaries(method_preds, gts)
    target_rows = build_diagnostic_rows(contexts, detail, counts, shortlist_counts, good_counts, good_shortlist_counts)
    image_rows = build_image_rows(contexts, method_preds, gts)
    recovery_rows = recovery_counts(method_preds, gts)
    diagnostic_rows = diagnostic_counts(target_rows)

    write_csv(OUTPUT_DIR / "oracle187_method_summary.csv", method_rows, union_fieldnames(method_rows))
    write_csv(OUTPUT_DIR / "oracle187_target_diagnostic_rows.csv", target_rows, union_fieldnames(target_rows))
    write_csv(OUTPUT_DIR / "oracle187_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "oracle187_recovery_summary.csv", recovery_rows, union_fieldnames(recovery_rows))
    write_csv(OUTPUT_DIR / "oracle187_diagnostic_summary.csv", diagnostic_rows, union_fieldnames(diagnostic_rows))
    write_summary(method_rows, recovery_rows, diagnostic_rows)

    print("\n===== ORACLE GLOBAL CHECKPOINT186 =====")
    for row in method_rows:
        print_required_metric_block(row["method"], row)
    print("\nRecuperacao das falhas/severos do 186:")
    for row in recovery_rows:
        print(
            f"  {row['method']}: falhas {row['failures_correctable_to_le5']} / "
            f"{row['baseline_failures_gt5']} | severos {row['severes_correctable_to_le8']} / "
            f"{row['baseline_severes_gt8']}"
        )
    print("\nDiagnostico:")
    for row in diagnostic_rows:
        print(f"  {row['diagnostic']}: {row['count']}")
    print(f"\nResumo: {OUTPUT_DIR / 'oracle187_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
