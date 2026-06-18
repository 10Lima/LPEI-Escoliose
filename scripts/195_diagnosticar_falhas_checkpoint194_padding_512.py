import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import TARGETS, round_safe, safe_float, safe_int


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT194_DIR = Path(os.environ.get(
    "SPINAL_195_CHECKPOINT194_DIR",
    str(ACTIVE_DIR / "checkpoint194_checkpoint191_pt193_v1"),
))
ORACLE187_DIR = Path(os.environ.get(
    "SPINAL_195_ORACLE187_DIR",
    str(ACTIVE_DIR / "oracle_global_checkpoint186_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_195_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "diagnosticar_falhas_checkpoint194_v1"),
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


def oracle_index(rows):
    return {(row["file_id"], row["target"]): row for row in rows}


def severity_label(mae3):
    mae3 = safe_float(mae3)
    if mae3 <= 5.0:
        return "pass_le5"
    if mae3 <= 6.0:
        return "fail_5_6"
    if mae3 <= 8.0:
        return "fail_6_8"
    if mae3 <= 10.0:
        return "fail_8_10"
    return "fail_gt10"


def error_bin(value):
    value = safe_float(value)
    if value <= 3.0:
        return "le3"
    if value <= 5.0:
        return "3_5"
    if value <= 8.0:
        return "5_8"
    if value <= 10.0:
        return "8_10"
    return "gt10"


def dominant_targets(errors):
    max_error = max(errors.values())
    return [target for target, value in errors.items() if abs(value - max_error) <= 1e-6]


def infer_target_cause(current_error, oracle):
    current_error = safe_float(current_error)
    if current_error <= 5.0:
        return "target_ok"
    good_shortlist = safe_int(oracle.get("good_candidate_shortlist_le5", 0)) == 1
    good_total = safe_int(oracle.get("good_candidate_total_le5", 0)) == 1
    enters_not_chosen = safe_int(oracle.get("candidate_enters_shortlist_not_chosen", 0)) == 1
    total_count = safe_int(oracle.get("candidate_count_total", 0))
    if good_shortlist and enters_not_chosen:
        return "selector_gate"
    if good_shortlist:
        return "shortlist_selector_choice"
    if good_total:
        return "shortlist_gap"
    if total_count > 0:
        return "candidate_generation_or_geometry"
    return "not_exported_or_geometry"


def build_audit_rows(image_rows, target_rows, oracle_rows):
    target_by_file = target_index(target_rows)
    oracle_by_key = oracle_index(oracle_rows)
    image_audit = []
    target_audit = []
    for image in image_rows:
        file_id = image["file_id"]
        errors = {
            "PT": safe_float(image["pt_abs_error"]),
            "MT": safe_float(image["mt_abs_error"]),
            "TL_L": safe_float(image["tll_abs_error"]),
        }
        dominant = dominant_targets(errors)
        failed_targets = [target for target, value in errors.items() if value > 5.0]
        severe_targets = [target for target, value in errors.items() if value > 8.0]
        target_causes = []
        oracle_total_good_failed = []
        oracle_shortlist_good_failed = []
        for target in TARGETS:
            oracle = oracle_by_key.get((file_id, target), {})
            cause = infer_target_cause(errors[target], oracle)
            if errors[target] > 5.0:
                target_causes.append(cause)
                if safe_int(oracle.get("good_candidate_total_le5", 0)) == 1:
                    oracle_total_good_failed.append(target)
                if safe_int(oracle.get("good_candidate_shortlist_le5", 0)) == 1:
                    oracle_shortlist_good_failed.append(target)

        if len(failed_targets) >= 2 and oracle_shortlist_good_failed:
            main_cause = "multi_target_selector_or_shortlist"
        elif len(failed_targets) >= 2 and oracle_total_good_failed:
            main_cause = "multi_target_shortlist_gap"
        elif len(failed_targets) >= 2:
            main_cause = "multi_target_generation_geometry"
        elif any(cause in {"selector_gate", "shortlist_selector_choice"} for cause in target_causes):
            main_cause = "selector_gate"
        elif any(cause == "shortlist_gap" for cause in target_causes):
            main_cause = "shortlist_gap"
        elif any("geometry" in cause for cause in target_causes):
            main_cause = "candidate_generation_or_geometry"
        else:
            main_cause = "borderline_or_mixed"

        image_audit.append({
            "file_id": file_id,
            "raw_file_id": image.get("raw_file_id", ""),
            "mae3": image["mae3"],
            "fail194": int(safe_float(image["mae3"]) > 5.0),
            "severe194": int(safe_float(image["mae3"]) > 8.0),
            "severity": severity_label(image["mae3"]),
            "dominant_targets": "+".join(dominant),
            "failed_targets_gt5": "+".join(failed_targets),
            "severe_targets_gt8": "+".join(severe_targets),
            "pt_abs_error": image["pt_abs_error"],
            "mt_abs_error": image["mt_abs_error"],
            "tll_abs_error": image["tll_abs_error"],
            "pt_error_bin": error_bin(image["pt_abs_error"]),
            "mt_error_bin": error_bin(image["mt_abs_error"]),
            "tll_error_bin": error_bin(image["tll_abs_error"]),
            "pt_source194": image["pt_source194"],
            "mt_source194": image["mt_source194"],
            "tll_source194": image["tll_source194"],
            "changed_targets194": image.get("changed_targets194", ""),
            "oracle_total_good_failed_targets": "+".join(oracle_total_good_failed),
            "oracle_shortlist_good_failed_targets": "+".join(oracle_shortlist_good_failed),
            "main_cause": main_cause,
        })

        for target in TARGETS:
            target_row = target_by_file[file_id][target]
            oracle = oracle_by_key.get((file_id, target), {})
            current_error = safe_float(target_row["abs_error"])
            oracle_total_error = safe_float(oracle.get("oracle_total_abs_error", ""))
            oracle_short_error = safe_float(oracle.get("oracle_shortlist_abs_error", ""))
            target_audit.append({
                "file_id": file_id,
                "raw_file_id": image.get("raw_file_id", ""),
                "target": target,
                "gt": target_row["gt"],
                "estimated": target_row["estimated"],
                "abs_error": target_row["abs_error"],
                "source194": target_row["source194"],
                "image_mae3": image["mae3"],
                "image_fail194": int(safe_float(image["mae3"]) > 5.0),
                "image_severe194": int(safe_float(image["mae3"]) > 8.0),
                "target_is_dominant": int(target in dominant),
                "target_failed_gt5": int(current_error > 5.0),
                "target_severe_gt8": int(current_error > 8.0),
                "target_error_bin": error_bin(current_error),
                "oracle_total_abs_error": round_safe(oracle_total_error),
                "oracle_shortlist_abs_error": round_safe(oracle_short_error),
                "oracle_total_gain_vs_194": round_safe(current_error - oracle_total_error if np.isfinite(oracle_total_error) else np.nan),
                "oracle_shortlist_gain_vs_194": round_safe(current_error - oracle_short_error if np.isfinite(oracle_short_error) else np.nan),
                "good_candidate_total_le5": oracle.get("good_candidate_total_le5", ""),
                "good_candidate_shortlist_le5": oracle.get("good_candidate_shortlist_le5", ""),
                "candidate_count_total": oracle.get("candidate_count_total", ""),
                "candidate_count_shortlist_rank_le25": oracle.get("candidate_count_shortlist_rank_le25", ""),
                "candidate_enters_shortlist_not_chosen": oracle.get("candidate_enters_shortlist_not_chosen", ""),
                "inferred_cause": infer_target_cause(current_error, oracle),
            })
    return image_audit, target_audit


def summarize_counts(image_rows, target_rows):
    rows = []
    fail_images = [row for row in image_rows if safe_int(row["fail194"]) == 1]
    severe_images = [row for row in image_rows if safe_int(row["severe194"]) == 1]
    failed_targets = [row for row in target_rows if safe_int(row["target_failed_gt5"]) == 1]
    severe_targets = [row for row in target_rows if safe_int(row["target_severe_gt8"]) == 1]
    rows.extend([
        {"metric": "fail_images_gt5", "group": "all", "count": len(fail_images)},
        {"metric": "severe_images_gt8", "group": "all", "count": len(severe_images)},
        {"metric": "failed_targets_gt5", "group": "all", "count": len(failed_targets)},
        {"metric": "severe_targets_gt8", "group": "all", "count": len(severe_targets)},
    ])
    for label, items, col in [
        ("fail_images_by_dominant", fail_images, "dominant_targets"),
        ("severe_images_by_dominant", severe_images, "dominant_targets"),
        ("fail_images_by_main_cause", fail_images, "main_cause"),
        ("severe_images_by_main_cause", severe_images, "main_cause"),
        ("fail_images_by_failed_targets", fail_images, "failed_targets_gt5"),
        ("severe_images_by_failed_targets", severe_images, "severe_targets_gt8"),
    ]:
        for value, count in Counter(row[col] for row in items).most_common():
            rows.append({"metric": label, "group": value, "count": count})
    for label, items in [("failed_targets_by_target", failed_targets), ("severe_targets_by_target", severe_targets)]:
        for value, count in Counter(row["target"] for row in items).most_common():
            rows.append({"metric": label, "group": value, "count": count})
    for label, items in [("failed_targets_by_cause", failed_targets), ("severe_targets_by_cause", severe_targets)]:
        for value, count in Counter(row["inferred_cause"] for row in items).most_common():
            rows.append({"metric": label, "group": value, "count": count})
    for target in TARGETS:
        target_failed = [row for row in failed_targets if row["target"] == target]
        total_good = sum(safe_int(row["good_candidate_total_le5"]) == 1 for row in target_failed)
        shortlist_good = sum(safe_int(row["good_candidate_shortlist_le5"]) == 1 for row in target_failed)
        rows.append({"metric": "failed_targets_oracle_total_le5", "group": target, "count": total_good})
        rows.append({"metric": "failed_targets_oracle_shortlist_le5", "group": target, "count": shortlist_good})
    return rows


def priority_rows(target_rows):
    failed = [row for row in target_rows if safe_int(row["target_failed_gt5"]) == 1]
    scored = []
    for row in failed:
        gain_total = safe_float(row["oracle_total_gain_vs_194"])
        gain_short = safe_float(row["oracle_shortlist_gain_vs_194"])
        score = 0.0
        score += max(gain_total if np.isfinite(gain_total) else 0.0, 0.0)
        score += 5.0 if safe_int(row["image_severe194"]) else 0.0
        score += 3.0 if safe_int(row["good_candidate_shortlist_le5"]) else 0.0
        score += 1.0 if np.isfinite(gain_short) and gain_short > 0.0 else 0.0
        scored.append({**row, "priority_score": round_safe(score)})
    return sorted(scored, key=lambda row: safe_float(row["priority_score"]), reverse=True)


def count_lookup(count_rows):
    return {(row["metric"], row["group"]): safe_int(row["count"]) for row in count_rows}


def write_summary(count_rows, priority):
    counts = count_lookup(count_rows)
    lines = [
        "# Diagnostico 195 falhas checkpoint194",
        "",
        "## Entradas",
        "",
        f"- Checkpoint194: `{CHECKPOINT194_DIR}`.",
        f"- Oracle187: `{ORACLE187_DIR}`.",
        "",
        "## Resumo",
        "",
        f"- Falhas >5: {counts.get(('fail_images_gt5', 'all'), 0)}.",
        f"- Severos >8: {counts.get(('severe_images_gt8', 'all'), 0)}.",
        f"- Targets falhados >5: {counts.get(('failed_targets_gt5', 'all'), 0)}.",
        f"- Targets severos >8: {counts.get(('severe_targets_gt8', 'all'), 0)}.",
        "",
        "## Falhas por target",
        "",
    ]
    for target in TARGETS:
        lines.append(
            f"- {target}: {counts.get(('failed_targets_by_target', target), 0)} falhas target; "
            f"oracle_total<=5 em {counts.get(('failed_targets_oracle_total_le5', target), 0)}; "
            f"oracle_shortlist<=5 em {counts.get(('failed_targets_oracle_shortlist_le5', target), 0)}."
        )
    lines.extend(["", "## Dominancia em imagens falhadas", ""])
    for row in count_rows:
        if row["metric"] == "fail_images_by_dominant":
            lines.append(f"- {row['group']}: {row['count']}")
    lines.extend(["", "## Causas principais em imagens falhadas", ""])
    for row in count_rows:
        if row["metric"] == "fail_images_by_main_cause":
            lines.append(f"- {row['group']}: {row['count']}")
    lines.extend(["", "## Top prioridades", ""])
    for row in priority[:15]:
        lines.append(
            f"- {row['file_id']} {row['target']}: erro {row['abs_error']}, "
            f"oracle_total {row['oracle_total_abs_error']}, causa {row['inferred_cause']}, "
            f"score {row['priority_score']}."
        )
    lines.extend(["", "## Leitura", ""])
    pt_fail = counts.get(("failed_targets_by_target", "PT"), 0)
    mt_fail = counts.get(("failed_targets_by_target", "MT"), 0)
    tll_fail = counts.get(("failed_targets_by_target", "TL_L"), 0)
    if tll_fail >= max(pt_fail, mt_fail):
        lines.append("- TL/L e a maior frente restante por contagem de targets falhados.")
    elif pt_fail >= max(mt_fail, tll_fail):
        lines.append("- PT e a maior frente restante por contagem de targets falhados.")
    else:
        lines.append("- MT continua relevante, mas ja foi parcialmente explorado no 190.")
    lines.append("- Priorizar candidatos com oracle_shortlist<=5 antes de atacar lacunas de geracao.")
    (OUTPUT_DIR / "diagnostico195_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    image_rows = read_csv(CHECKPOINT194_DIR / "checkpoint194_image_rows.csv")
    target_rows = read_csv(CHECKPOINT194_DIR / "checkpoint194_target_rows.csv")
    oracle_rows = read_csv(ORACLE187_DIR / "oracle187_target_diagnostic_rows.csv")
    image_audit, target_audit = build_audit_rows(image_rows, target_rows, oracle_rows)
    count_rows = summarize_counts(image_audit, target_audit)
    priorities = priority_rows(target_audit)

    write_csv(OUTPUT_DIR / "diagnostico195_image_rows.csv", image_audit, union_fieldnames(image_audit))
    write_csv(OUTPUT_DIR / "diagnostico195_target_rows.csv", target_audit, union_fieldnames(target_audit))
    write_csv(OUTPUT_DIR / "diagnostico195_count_summary.csv", count_rows, union_fieldnames(count_rows))
    write_csv(OUTPUT_DIR / "diagnostico195_priority_targets.csv", priorities, union_fieldnames(priorities))
    write_summary(count_rows, priorities)

    print("\n===== DIAGNOSTICO195 FALHAS CHECKPOINT194 =====")
    for row in count_rows:
        if row["metric"] in {
            "fail_images_gt5", "severe_images_gt8", "failed_targets_gt5", "severe_targets_gt8",
            "failed_targets_by_target", "failed_targets_oracle_total_le5",
            "failed_targets_oracle_shortlist_le5",
        }:
            print(f"{row['metric']} | {row['group']}: {row['count']}")
    print(f"\nResumo: {OUTPUT_DIR / 'diagnostico195_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
