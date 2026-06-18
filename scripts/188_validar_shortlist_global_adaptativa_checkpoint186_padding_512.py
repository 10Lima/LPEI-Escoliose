import csv
import itertools
import os
from collections import defaultdict
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
)


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT186_DIR = Path(os.environ.get(
    "SPINAL_188_CHECKPOINT186_DIR",
    str(ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1"),
))
PT_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_188_PT_CANDIDATES_DIR",
    str(ACTIVE_DIR / "gate_pt_hard_negatives_checkpoint114_v1"),
))
MT_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_188_MT_CANDIDATES_DIR",
    str(ACTIVE_DIR / "validar_selector_mt_diversidade_checkpoint114_global_top1200_v1"),
))
TLL_OLD_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_188_TLL_OLD_CANDIDATES_DIR",
    str(ACTIVE_DIR / "gate_two_stage_hard_negatives_tll_v1"),
))
TLL159_DIR = Path(os.environ.get(
    "SPINAL_188_TLL159_DIR",
    str(ACTIVE_DIR / "gerar_candidatos_tll_checkpoint156_relevantes_v1"),
))
TLL175_DIR = Path(os.environ.get(
    "SPINAL_188_TLL175_DIR",
    str(ACTIVE_DIR / "gerar_candidatos_tll_remaining_selector173_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_188_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "validar_shortlist_global_adaptativa_checkpoint186_v1"),
))
MAX_FILES = safe_int(os.environ.get("SPINAL_188_MAX_FILES", "0"))

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
            "base": estimated,
            "base_errors": [abs(estimated[idx] - gt[idx]) for idx in range(3)],
            "sources186": {target: values[target].get("source186", "") for target in TARGETS},
        }
    return contexts


def first_score(row, cols):
    for col in cols:
        value = safe_float(row.get(col, ""))
        if np.isfinite(value):
            return value
    return np.nan


def specs():
    return [
        {
            "target": "PT",
            "path": PT_CANDIDATES_DIR / "gate123_oof_candidate_predictions.csv",
            "source": "old_PT",
            "rank_col": "rank",
            "score_cols": ["gate123_score", "predicted_good_score", "selection_score"],
            "method_col": "window_name",
        },
        {
            "target": "MT",
            "path": MT_CANDIDATES_DIR / "selector125_oof_candidate_predictions.csv",
            "source": "old_MT",
            "rank_col": "rank",
            "score_cols": ["predicted_good_score", "selection_score"],
            "method_col": "window_name",
        },
        {
            "target": "TL_L",
            "path": TLL_OLD_CANDIDATES_DIR / "gate114_oof_candidate_predictions.csv",
            "source": "old_TLL",
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


def stream_candidates(contexts):
    wanted = set(contexts.keys())
    groups = defaultdict(list)
    for spec in specs():
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"CSV de candidatos nao encontrado: {path}")
        print(f"A ler candidatos {spec['source']}: {path}", flush=True)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row.get("file_id", "")
                if file_id not in wanted:
                    continue
                target = spec["target"]
                idx = TARGETS.index(target)
                cobb = safe_float(row.get("cobb", ""))
                label_abs_error = safe_float(row.get("label_abs_error", ""))
                if not np.isfinite(cobb):
                    continue
                base = contexts[file_id]["base"][idx]
                groups[(file_id, target)].append({
                    "file_id": file_id,
                    "target": target,
                    "source": spec["source"],
                    "source_group": source_group(spec["source"]),
                    "method": row.get(spec["method_col"], "") if spec["method_col"] else spec["source"],
                    "rank": safe_int(row.get(spec["rank_col"], "")),
                    "score": first_score(row, spec["score_cols"]),
                    "cobb": cobb,
                    "label_abs_error": label_abs_error,
                    "jump": abs(cobb - base),
                    "signed_jump": cobb - base,
                    "base_cobb": base,
                })
    return groups


def source_group(source):
    if source == "old_TLL":
        return "old"
    if source in {"tll159", "tll175"}:
        return "new"
    return "old"


def regional_profiles():
    return {
        "PT": [
            {"id": "pt_off", "max_rank": 0, "max_jump": 0.0, "min_score": np.inf, "sources": "old"},
            {"id": "pt_r25_j8_s0", "max_rank": 25, "max_jump": 8.0, "min_score": 0.0, "sources": "old"},
            {"id": "pt_r100_j12_s005", "max_rank": 100, "max_jump": 12.0, "min_score": 0.05, "sources": "old"},
            {"id": "pt_r500_j16_s02", "max_rank": 500, "max_jump": 16.0, "min_score": 0.20, "sources": "old"},
            {"id": "pt_r1200_j25_s04", "max_rank": 1200, "max_jump": 25.0, "min_score": 0.40, "sources": "old"},
        ],
        "MT": [
            {"id": "mt_off", "max_rank": 0, "max_jump": 0.0, "min_score": np.inf, "sources": "old"},
            {"id": "mt_r25_j8_s04", "max_rank": 25, "max_jump": 8.0, "min_score": 0.40, "sources": "old"},
            {"id": "mt_r100_j12_s045", "max_rank": 100, "max_jump": 12.0, "min_score": 0.45, "sources": "old"},
            {"id": "mt_r500_j16_s05", "max_rank": 500, "max_jump": 16.0, "min_score": 0.50, "sources": "old"},
            {"id": "mt_r1200_j22_s055", "max_rank": 1200, "max_jump": 22.0, "min_score": 0.55, "sources": "old"},
        ],
        "TL_L": [
            {"id": "tll_off", "max_rank": 0, "max_jump": 0.0, "min_score": np.inf, "sources": "all"},
            {"id": "tll_old_r25_j8_s05", "max_rank": 25, "max_jump": 8.0, "min_score": 0.50, "sources": "old"},
            {"id": "tll_all_r100_j12_s0", "max_rank": 100, "max_jump": 12.0, "min_score": 0.0, "sources": "all"},
            {"id": "tll_all_r500_j16_s0", "max_rank": 500, "max_jump": 16.0, "min_score": 0.0, "sources": "all"},
            {"id": "tll_new_r2000_j25_s0", "max_rank": 2000, "max_jump": 25.0, "min_score": 0.0, "sources": "new"},
        ],
    }


def configs():
    profiles = regional_profiles()
    rows = []
    for pt, mt, tll, mode in itertools.product(
        profiles["PT"], profiles["MT"], profiles["TL_L"], ["rank", "score"]
    ):
        rows.append({
            "config_id": f"188__{pt['id']}__{mt['id']}__{tll['id']}__sel_{mode}",
            "selection_mode": mode,
            "PT": pt,
            "MT": mt,
            "TL_L": tll,
        })
    return rows


def candidate_passes(item, profile):
    if safe_int(profile["max_rank"]) <= 0:
        return False
    if safe_int(item["rank"]) > safe_int(profile["max_rank"]):
        return False
    if safe_float(item["jump"]) > safe_float(profile["max_jump"]):
        return False
    if safe_float(item["score"]) < safe_float(profile["min_score"]):
        return False
    sources = profile["sources"]
    if sources != "all" and item["source_group"] != sources:
        return False
    return True


def select_candidate(items, profile, mode):
    passed = [item for item in items if candidate_passes(item, profile)]
    if not passed:
        return None
    if mode == "rank":
        return min(passed, key=lambda item: (safe_int(item["rank"]), -safe_float(item["score"])))
    return max(passed, key=lambda item: (safe_float(item["score"]), -safe_int(item["rank"])))


def profile_oracle(items, profile):
    passed = [
        item for item in items
        if candidate_passes(item, profile) and np.isfinite(safe_float(item["label_abs_error"]))
    ]
    if not passed:
        return None
    return min(passed, key=lambda item: (safe_float(item["label_abs_error"]), safe_int(item["rank"])))


def build_selection_caches(candidates):
    profiles = regional_profiles()
    selected = {}
    oracle = {}
    for key, items in candidates.items():
        _, target = key
        for profile in profiles[target]:
            profile_id = profile["id"]
            oracle[(key, profile_id)] = profile_oracle(items, profile)
            for mode in ["rank", "score"]:
                selected[(key, profile_id, mode)] = select_candidate(items, profile, mode)
    return selected, oracle


def evaluate_config(config, contexts, selection_cache, oracle_cache):
    file_ids = sorted(contexts.keys())
    preds = []
    gts = []
    selected_cases = set()
    selected_targets = 0
    target_deltas = []
    profile_oracle_good = 0
    profile_oracle_fail_targets = 0
    fixed_failures_by_selected = 0
    fixed_severes_by_selected = 0

    for file_id in file_ids:
        context = contexts[file_id]
        est = list(context["base"])
        base_mae = float(np.mean(context["base_errors"]))
        for target in TARGETS:
            idx = TARGETS.index(target)
            profile = config[target]
            key = (file_id, target)
            oracle = oracle_cache.get((key, profile["id"]))
            if context["base_errors"][idx] > 5.0:
                profile_oracle_fail_targets += 1
                if oracle is not None and safe_float(oracle["label_abs_error"]) <= 5.0:
                    profile_oracle_good += 1
            item = selection_cache.get((key, profile["id"], config["selection_mode"]))
            if item is None:
                continue
            new_error = abs(safe_float(item["cobb"]) - context["gt"][idx])
            target_deltas.append(new_error - context["base_errors"][idx])
            est[idx] = safe_float(item["cobb"])
            selected_cases.add(file_id)
            selected_targets += 1
        mae = float(np.mean(np.abs(np.asarray(est) - np.asarray(context["gt"]))))
        fixed_failures_by_selected += int(base_mae > 5.0 and mae <= 5.0)
        fixed_severes_by_selected += int(base_mae > 8.0 and mae <= 8.0)
        preds.append(est)
        gts.append(context["gt"])

    summary = metric_summary_from_arrays(np.asarray(preds), np.asarray(gts), baseline_preds=base_array(contexts))
    summary.update({
        "config_id": config["config_id"],
        "selection_mode": config["selection_mode"],
        "pt_profile": config["PT"]["id"],
        "mt_profile": config["MT"]["id"],
        "tll_profile": config["TL_L"]["id"],
        "selected_cases": len(selected_cases),
        "selected_targets": selected_targets,
        "improved_targets": int(np.sum(np.asarray(target_deltas) < -1e-6)) if target_deltas else 0,
        "worsened_targets": int(np.sum(np.asarray(target_deltas) > 1e-6)) if target_deltas else 0,
        "fixed_failures_by_selected": fixed_failures_by_selected,
        "fixed_severes_by_selected": fixed_severes_by_selected,
        "profile_oracle_good_fail_targets_le5": profile_oracle_good,
        "profile_oracle_fail_targets_gt5": profile_oracle_fail_targets,
    })
    return summary


def base_array(contexts):
    return np.asarray([contexts[file_id]["base"] for file_id in sorted(contexts.keys())], dtype=float)


def gt_array(contexts):
    return np.asarray([contexts[file_id]["gt"] for file_id in sorted(contexts.keys())], dtype=float)


def sort_safe(summary, baseline):
    return (
        safe_int(summary["regressions_target_gt5"]) == 0,
        safe_int(summary["regressions_image_gt3"]) == 0,
        safe_int(summary["regressions_image_gt1"]) == 0,
        safe_int(summary["failures_gt5"]) <= safe_int(baseline["failures_gt5"]),
        safe_int(summary["severe_gt8"]) <= safe_int(baseline["severe_gt8"]),
        -safe_int(summary["severe_gt8"]),
        -safe_int(summary["failures_gt5"]),
        -safe_float(summary["mae3"]),
        safe_float(summary["within_5"]),
    )


def sort_balanced(summary, baseline):
    return (
        safe_int(summary["regressions_target_gt5"]) == 0,
        safe_int(summary["regressions_image_gt3"]) == 0,
        safe_int(summary["failures_gt5"]) <= safe_int(baseline["failures_gt5"]),
        -safe_int(summary["severe_gt8"]),
        -safe_int(summary["failures_gt5"]),
        -safe_int(summary["regressions_image_gt1"]),
        -safe_float(summary["mae3"]),
    )


def build_detail_rows(config, contexts, selection_cache):
    file_ids = sorted(contexts.keys())
    image_rows = []
    target_rows = []
    selected_rows = []
    for file_id in file_ids:
        context = contexts[file_id]
        est = list(context["base"])
        sources = dict(context["sources186"])
        changed = []
        for target in TARGETS:
            idx = TARGETS.index(target)
            item = selection_cache.get(((file_id, target), config[target]["id"], config["selection_mode"]))
            if item is None:
                continue
            new_error = abs(safe_float(item["cobb"]) - context["gt"][idx])
            selected_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "profile": config[target]["id"],
                "rank": item["rank"],
                "source": item["source"],
                "method": item["method"],
                "score": round_safe(item["score"]),
                "selected_cobb": round_safe(item["cobb"]),
                "gt": round_safe(context["gt"][idx]),
                "base_cobb186": round_safe(context["base"][idx]),
                "base_abs_error186": round_safe(context["base_errors"][idx]),
                "selected_abs_error": round_safe(new_error),
                "delta_abs_error": round_safe(new_error - context["base_errors"][idx]),
                "jump": round_safe(item["jump"]),
                "config_id": config["config_id"],
            })
            est[idx] = safe_float(item["cobb"])
            sources[target] = f"adaptive188_{item['source']}"
            changed.append(target)
        errors = np.abs(np.asarray(est) - np.asarray(context["gt"]))
        image_rows.append({
            "file_id": file_id,
            "raw_file_id": context["raw_file_id"],
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(np.mean(errors)),
            "within_5": int(np.mean(errors) <= 5.0),
            "pt_source188": sources["PT"],
            "mt_source188": sources["MT"],
            "tll_source188": sources["TL_L"],
            "changed_targets188": "+".join(changed),
        })
        for idx, target in enumerate(TARGETS):
            target_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "gt": round_safe(context["gt"][idx]),
                "estimated": round_safe(est[idx]),
                "abs_error": round_safe(errors[idx]),
                "source188": sources[target],
            })
    return selected_rows, image_rows, target_rows


def compact_summary(row):
    keys = [
        "config_id", "mae3", "rmse3", "within_5", "within_10", "within_15",
        "failures_gt5", "severe_gt8", "regressions_target_gt5",
        "regressions_image_gt1", "regressions_image_gt3", "selected_cases",
        "selected_targets", "fixed_failures_by_selected", "fixed_severes_by_selected",
        "max_cobb_smape_article", "pt_smape_article", "mt_smape_article",
        "tll_smape_article", "agg3_smape_article",
    ]
    return {key: row.get(key, "") for key in keys}


def write_summary(baseline, best_safe, best_balanced):
    lines = [
        "# Shortlist global adaptativa checkpoint186",
        "",
        "## Escopo",
        "",
        "- Valida shortlist global com regras separadas para PT, MT e TL/L.",
        "- A selecao usa apenas rank, score, jump e fonte do candidato.",
        "- GT e usado apenas para metricas e cobertura oracle da shortlist.",
        "- Fallback obrigatorio: checkpoint186.",
        "",
        "## Baseline checkpoint186",
        "",
    ]
    for key, value in compact_summary({"config_id": "checkpoint186", **baseline}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor safe 188", ""])
    for key, value in compact_summary(best_safe).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor balanced 188", ""])
    for key, value in compact_summary(best_balanced).items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Leitura",
        "",
        "- Se o perfil oracle da shortlist tiver muita cobertura mas a selecao real nao melhorar, o gargalo e selector/gate.",
        "- Se a cobertura do perfil continuar baixa, o gargalo e geracao/shortlist/bandas.",
        "- Este script nao promove checkpoint.",
    ])
    (OUTPUT_DIR / "shortlist188_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    contexts = load_contexts()
    candidates = stream_candidates(contexts)
    selection_cache, oracle_cache = build_selection_caches(candidates)
    baseline = metric_summary_from_arrays(base_array(contexts), gt_array(contexts))

    rows = []
    best_safe = None
    best_balanced = None
    best_safe_config = None
    best_balanced_config = None
    for config in configs():
        summary = evaluate_config(config, contexts, selection_cache, oracle_cache)
        rows.append(summary)
        safe_key = sort_safe(summary, baseline)
        balanced_key = sort_balanced(summary, baseline)
        if best_safe is None or safe_key > best_safe[0]:
            best_safe = (safe_key, summary)
            best_safe_config = config
        if best_balanced is None or balanced_key > best_balanced[0]:
            best_balanced = (balanced_key, summary)
            best_balanced_config = config

    best_safe_summary = best_safe[1]
    best_balanced_summary = best_balanced[1]
    selected_rows, image_rows, target_rows = build_detail_rows(best_safe_config, contexts, selection_cache)
    balanced_selected, balanced_images, balanced_targets = build_detail_rows(best_balanced_config, contexts, selection_cache)

    write_csv(OUTPUT_DIR / "shortlist188_config_summary.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "shortlist188_best_safe_summary.csv", [best_safe_summary], union_fieldnames([best_safe_summary]))
    write_csv(OUTPUT_DIR / "shortlist188_best_balanced_summary.csv", [best_balanced_summary], union_fieldnames([best_balanced_summary]))
    write_csv(OUTPUT_DIR / "shortlist188_best_safe_selected_cases.csv", selected_rows, union_fieldnames(selected_rows) if selected_rows else ["file_id"])
    write_csv(OUTPUT_DIR / "shortlist188_best_safe_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "shortlist188_best_safe_target_rows.csv", target_rows, union_fieldnames(target_rows))
    write_csv(OUTPUT_DIR / "shortlist188_best_balanced_selected_cases.csv", balanced_selected, union_fieldnames(balanced_selected) if balanced_selected else ["file_id"])
    write_csv(OUTPUT_DIR / "shortlist188_best_balanced_image_rows.csv", balanced_images, union_fieldnames(balanced_images))
    write_csv(OUTPUT_DIR / "shortlist188_best_balanced_target_rows.csv", balanced_targets, union_fieldnames(balanced_targets))
    write_summary(baseline, best_safe_summary, best_balanced_summary)

    print("\n===== SHORTLIST188 GLOBAL ADAPTATIVA =====")
    print_required_metric_block("checkpoint186", baseline)
    print_required_metric_block("melhor safe 188", best_safe_summary)
    print_required_metric_block("melhor balanced 188", best_balanced_summary)
    print(f"\nResumo: {OUTPUT_DIR / 'shortlist188_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
