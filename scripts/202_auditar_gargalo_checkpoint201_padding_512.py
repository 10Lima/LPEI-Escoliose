import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import TARGETS, metric_summary_from_arrays, round_safe, safe_float, safe_int


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT186_DIR = ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1"
CHECKPOINT191_DIR = ACTIVE_DIR / "checkpoint191_checkpoint186_mt190_v1"
CHECKPOINT198_DIR = ACTIVE_DIR / "checkpoint198_checkpoint194_tll197_v1"
CHECKPOINT201_DIR = ACTIVE_DIR / "checkpoint201_checkpoint198_pt200_v1"
DIAG199_DIR = ACTIVE_DIR / "diagnosticar_falhas_checkpoint198_v1"
ORACLE187_DIR = ACTIVE_DIR / "oracle_global_checkpoint186_v1"
AUDIT183_DIR = ACTIVE_DIR / "auditoria_critica_severos_checkpoint182_v2"
SEVERE180_DIR = ACTIVE_DIR / "oracle_visual_severas_selector179_v2"
GEOMETRY177_DIR = ACTIVE_DIR / "quantificar_geometria_centerline_selector173_v1"
OUTPUT_DIR = ACTIVE_DIR / "auditar_gargalo_checkpoint201_v1"


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


def image_index(rows):
    return {row["file_id"]: row for row in rows}


def oracle_index(rows):
    return {(row["file_id"], row["target"]): row for row in rows}


def split_targets(value):
    if not value:
        return []
    return [item for item in value.split("+") if item]


def read_checkpoint_image_rows(label):
    if label == "186":
        return read_csv(CHECKPOINT186_DIR / "selector186_best_image_rows.csv")
    if label == "191":
        return read_csv(CHECKPOINT191_DIR / "checkpoint191_image_rows.csv")
    if label == "198":
        return read_csv(CHECKPOINT198_DIR / "checkpoint198_image_rows.csv")
    if label == "201":
        return read_csv(CHECKPOINT201_DIR / "checkpoint201_image_rows.csv")
    raise ValueError(label)


def read_checkpoint_target_rows(label):
    if label == "186":
        return read_csv(CHECKPOINT186_DIR / "selector186_best_target_rows.csv")
    if label == "191":
        return read_csv(CHECKPOINT191_DIR / "checkpoint191_target_rows.csv")
    if label == "198":
        return read_csv(CHECKPOINT198_DIR / "checkpoint198_target_rows.csv")
    if label == "201":
        return read_csv(CHECKPOINT201_DIR / "checkpoint201_target_rows.csv")
    raise ValueError(label)


def arrays_from_targets(rows):
    by_file = target_index(rows)
    preds = []
    gts = []
    for file_id in sorted(by_file):
        preds.append([safe_float(by_file[file_id][target]["estimated"]) for target in TARGETS])
        gts.append([safe_float(by_file[file_id][target]["gt"]) for target in TARGETS])
    return np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)


def error_triplet_from_image(row):
    return {
        "PT": safe_float(row["pt_abs_error"]),
        "MT": safe_float(row["mt_abs_error"]),
        "TL_L": safe_float(row["tll_abs_error"]),
    }


def dominant_targets(errors):
    max_error = max(errors.values())
    return [target for target, value in errors.items() if abs(value - max_error) <= 1e-6]


def classify_target_bottleneck(current_error, oracle):
    if current_error <= 5.0:
        return "target_ok"
    good_total = safe_int(oracle.get("good_candidate_total_le5", 0)) == 1
    good_short = safe_int(oracle.get("good_candidate_shortlist_le5", 0)) == 1
    enters_not_chosen = safe_int(oracle.get("candidate_enters_shortlist_not_chosen", 0)) == 1
    total_count = safe_int(oracle.get("candidate_count_total", 0))
    if good_short and enters_not_chosen:
        return "selector_gate"
    if good_short:
        return "candidate_exists_not_chosen"
    if good_total:
        return "shortlist_or_anatomical_band"
    if total_count > 0:
        return "candidate_generation_or_centerline"
    return "lack_of_candidates"


def load_optional_indexes():
    audit183 = {}
    if (AUDIT183_DIR / "audit183_severe_rows.csv").exists():
        audit183 = image_index(read_csv(AUDIT183_DIR / "audit183_severe_rows.csv"))
    severe180 = {}
    if (SEVERE180_DIR / "severe180_rows.csv").exists():
        severe180 = image_index(read_csv(SEVERE180_DIR / "severe180_rows.csv"))
    geometry177 = {}
    if (GEOMETRY177_DIR / "geometry177_all_rows.csv").exists():
        geometry177 = image_index(read_csv(GEOMETRY177_DIR / "geometry177_all_rows.csv"))
    return audit183, severe180, geometry177


def geometry_flags(file_id, audit183, geometry177):
    source = audit183.get(file_id) or geometry177.get(file_id) or {}
    centerline_unstable = safe_int(source.get("centerline_unstable_flag", 0)) == 1
    weak_mask = safe_int(source.get("weak_mask_flag", 0)) == 1
    wrong_band = safe_int(source.get("wrong_anatomical_band_flag", 0)) == 1
    angle_turns = safe_int(source.get("angle_turns_gt20", 0))
    angle_range = safe_float(source.get("angle_range", ""))
    y_coverage = safe_float(source.get("centerline_y_coverage", ""))
    mask_area = safe_float(source.get("pred_mask_area_pct", ""))
    if np.isfinite(angle_range) and angle_range >= 120.0:
        centerline_unstable = True
    if angle_turns >= 2:
        centerline_unstable = True
    if np.isfinite(y_coverage) and y_coverage < 0.65:
        centerline_unstable = True
    if np.isfinite(mask_area) and (mask_area < 0.5 or mask_area > 5.0):
        weak_mask = True
    return {
        "centerline_instavel": int(centerline_unstable),
        "mascara_segmentacao_suspeita": int(weak_mask),
        "problema_banda_anatomica_flag": int(wrong_band),
        "centerline_y_coverage": round_safe(y_coverage),
        "angle_range": round_safe(angle_range),
        "angle_turns_gt20": angle_turns,
        "pred_mask_area_pct": round_safe(mask_area),
    }


def severe_rows_201():
    image_rows = read_checkpoint_image_rows("201")
    target_rows = read_checkpoint_target_rows("201")
    target_by_file = target_index(target_rows)
    oracle_by_key = oracle_index(read_csv(ORACLE187_DIR / "oracle187_target_diagnostic_rows.csv"))
    audit183, severe180, geometry177 = load_optional_indexes()
    rows = []
    target_bottlenecks = []
    for image in image_rows:
        mae3 = safe_float(image["mae3"])
        if mae3 <= 8.0:
            continue
        file_id = image["file_id"]
        errors = error_triplet_from_image(image)
        dominant = dominant_targets(errors)
        failed = [target for target, value in errors.items() if value > 5.0]
        severe_targets = [target for target, value in errors.items() if value > 8.0]
        bottlenecks = []
        target_cause_map = {}
        for target in TARGETS:
            oracle = oracle_by_key.get((file_id, target), {})
            current_error = errors[target]
            cause = classify_target_bottleneck(current_error, oracle)
            target_cause_map[target] = cause
            if current_error > 5.0:
                bottlenecks.append(cause)
                target_bottlenecks.append({
                    "file_id": file_id,
                    "target": target,
                    "current_abs_error": round_safe(current_error),
                    "current_failed_gt5": int(current_error > 5.0),
                    "current_severe_gt8": int(current_error > 8.0),
                    "target_is_dominant": int(target in dominant),
                    "bottleneck": cause,
                    "oracle_total_abs_error": round_safe(oracle.get("oracle_total_abs_error", "")),
                    "oracle_shortlist_abs_error": round_safe(oracle.get("oracle_shortlist_abs_error", "")),
                    "good_candidate_total_le5": oracle.get("good_candidate_total_le5", ""),
                    "good_candidate_shortlist_le5": oracle.get("good_candidate_shortlist_le5", ""),
                    "candidate_count_total": oracle.get("candidate_count_total", ""),
                    "candidate_count_shortlist_rank_le25": oracle.get("candidate_count_shortlist_rank_le25", ""),
                    "candidate_enters_shortlist_not_chosen": oracle.get("candidate_enters_shortlist_not_chosen", ""),
                })
        flags = geometry_flags(file_id, audit183, geometry177)
        categories = set()
        if len(failed) >= 2:
            categories.add("multi-target")
        if flags["centerline_instavel"]:
            categories.add("centerline instavel")
        if flags["mascara_segmentacao_suspeita"]:
            categories.add("mascara/segmentacao problematica")
        if flags["problema_banda_anatomica_flag"] or "shortlist_or_anatomical_band" in bottlenecks:
            categories.add("problema de banda anatomica")
        if "lack_of_candidates" in bottlenecks:
            categories.add("falta de candidatos")
        if "candidate_generation_or_centerline" in bottlenecks:
            categories.add("geracao de candidatos/centerline")
        if "selector_gate" in bottlenecks or "candidate_exists_not_chosen" in bottlenecks:
            categories.add("candidato existe mas nao e escolhido")
        if any(value > 15.0 for value in errors.values()):
            categories.add("caso severo raro")
        if len(failed) >= 2 and not categories:
            categories.add("geometria estrutural dificil")
        if severe180.get(file_id, {}).get("severe_class180"):
            categories.add(severe180[file_id]["severe_class180"])

        rows.append({
            "file_id": file_id,
            "raw_file_id": image.get("raw_file_id", ""),
            "mae3": image["mae3"],
            "pt_abs_error": image["pt_abs_error"],
            "mt_abs_error": image["mt_abs_error"],
            "tll_abs_error": image["tll_abs_error"],
            "dominant_targets": "+".join(dominant),
            "failed_targets_gt5": "+".join(failed),
            "severe_targets_gt8": "+".join(severe_targets),
            "multi_target": int(len(failed) >= 2),
            "pt_dominant": int("PT" in dominant),
            "tll_dominant": int("TL_L" in dominant),
            "mt_dominant": int("MT" in dominant),
            "pt_cause": target_cause_map["PT"],
            "mt_cause": target_cause_map["MT"],
            "tll_cause": target_cause_map["TL_L"],
            "classes": "; ".join(sorted(categories)),
            **flags,
        })
    return rows, target_bottlenecks


def checkpoint_progress_rows():
    checkpoints = ["186", "191", "198", "201"]
    severe_sets = {}
    rows = []
    baseline_preds = None
    baseline_gts = None
    for label in checkpoints:
        target_rows = read_checkpoint_target_rows(label)
        preds, gts = arrays_from_targets(target_rows)
        if label == "186":
            baseline_preds = preds
            baseline_gts = gts
            summary = metric_summary_from_arrays(preds, gts)
        else:
            summary = metric_summary_from_arrays(preds, gts, baseline_preds=baseline_preds)
        image_rows = read_checkpoint_image_rows(label)
        severe_sets[label] = {row["file_id"] for row in image_rows if safe_float(row["mae3"]) > 8.0}
        fail_sets = {row["file_id"] for row in image_rows if safe_float(row["mae3"]) > 5.0}
        rows.append({
            "checkpoint": label,
            "mae3": summary["mae3"],
            "within_5": summary["within_5"],
            "failures_gt5": summary["failures_gt5"],
            "severe_gt8": summary["severe_gt8"],
            "max_cobb_smape_article": summary["max_cobb_smape_article"],
            "pt_smape_article": summary["pt_smape_article"],
            "mt_smape_article": summary["mt_smape_article"],
            "tll_smape_article": summary["tll_smape_article"],
            "agg3_smape_article": summary["agg3_smape_article"],
            "fail_set_count": len(fail_sets),
            "severe_set_count": len(severe_sets[label]),
        })
    overlap_rows = []
    current = severe_sets["201"]
    for label in ["186", "191", "198"]:
        overlap_rows.append({
            "comparison": f"201_vs_{label}",
            "severe201_count": len(current),
            f"severe{label}_count": len(severe_sets[label]),
            "same_severe_count": len(current & severe_sets[label]),
            "new_in_201_vs_ref": len(current - severe_sets[label]),
            "fixed_from_ref": len(severe_sets[label] - current),
        })
    return rows, overlap_rows


def current_target_rows():
    image_rows = read_checkpoint_image_rows("201")
    target_rows = read_checkpoint_target_rows("201")
    oracle_by_key = oracle_index(read_csv(ORACLE187_DIR / "oracle187_target_diagnostic_rows.csv"))
    target_audit = []
    for row in target_rows:
        file_id = row["file_id"]
        target = row["target"]
        current_error = safe_float(row["abs_error"])
        oracle = oracle_by_key.get((file_id, target), {})
        target_audit.append({
            "file_id": file_id,
            "target": target,
            "current_abs_error": round_safe(current_error),
            "current_failed_gt5": int(current_error > 5.0),
            "current_severe_gt8": int(current_error > 8.0),
            "bottleneck": classify_target_bottleneck(current_error, oracle),
            "oracle_total_abs_error": round_safe(oracle.get("oracle_total_abs_error", "")),
            "oracle_shortlist_abs_error": round_safe(oracle.get("oracle_shortlist_abs_error", "")),
            "good_candidate_total_le5": oracle.get("good_candidate_total_le5", ""),
            "good_candidate_shortlist_le5": oracle.get("good_candidate_shortlist_le5", ""),
            "candidate_count_total": oracle.get("candidate_count_total", ""),
            "candidate_count_shortlist_rank_le25": oracle.get("candidate_count_shortlist_rank_le25", ""),
        })
    return target_audit


def bottleneck_summary(target_audit, severe_target_audit):
    rows = []
    for scope, items in [
        ("all_failed_targets", [row for row in target_audit if safe_int(row["current_failed_gt5"]) == 1]),
        ("severe_targets", [row for row in target_audit if safe_int(row["current_severe_gt8"]) == 1]),
        ("targets_in_severe_images", severe_target_audit),
    ]:
        rows.append({"scope": scope, "bottleneck": "total", "count": len(items)})
        for bottleneck, count in Counter(row["bottleneck"] for row in items).most_common():
            rows.append({"scope": scope, "bottleneck": bottleneck, "count": count})
        for target in TARGETS:
            target_items = [row for row in items if row["target"] == target]
            rows.append({"scope": scope, "bottleneck": f"target_{target}", "count": len(target_items)})
    return rows


def oracle_ceiling_rows():
    target_rows = read_checkpoint_target_rows("201")
    by_file = target_index(target_rows)
    oracle_by_key = oracle_index(read_csv(ORACLE187_DIR / "oracle187_target_diagnostic_rows.csv"))
    rows = []
    current_abs = []
    total_abs = []
    shortlist_abs = []
    gts = []
    # Use GT values only to keep metric_summary shape and SMAPE denominators coherent.
    for file_id in sorted(by_file):
        current = []
        total = []
        shortlist = []
        gt = []
        for target in TARGETS:
            row = by_file[file_id][target]
            oracle = oracle_by_key.get((file_id, target), {})
            cur_error = safe_float(row["abs_error"])
            total_error = safe_float(oracle.get("oracle_total_abs_error", ""))
            short_error = safe_float(oracle.get("oracle_shortlist_abs_error", ""))
            current.append(cur_error)
            total.append(total_error if np.isfinite(total_error) else cur_error)
            shortlist.append(short_error if np.isfinite(short_error) else cur_error)
            gt.append(0.0)
        current_abs.append(current)
        total_abs.append(total)
        shortlist_abs.append(shortlist)
        gts.append(gt)
    for name, arr in [
        ("checkpoint201_abs_error", np.asarray(current_abs, dtype=float)),
        ("oracle_total_abs_error_ceiling", np.asarray(total_abs, dtype=float)),
        ("oracle_shortlist_abs_error_ceiling", np.asarray(shortlist_abs, dtype=float)),
    ]:
        mae3 = np.mean(arr, axis=1)
        rows.append({
            "method": name,
            "mae3_abs_error_only": round_safe(float(np.mean(mae3))),
            "within_5_abs_error_only": round_safe(float(np.mean(mae3 <= 5.0) * 100.0)),
            "failures_gt5_abs_error_only": int(np.sum(mae3 > 5.0)),
            "severe_gt8_abs_error_only": int(np.sum(mae3 > 8.0)),
        })
    return rows


def training_rows(severe_rows, target_audit, oracle_rows):
    train_count = ""
    val_count = ""
    train_path = PROCESSED_DIR / "splits" / "train.json"
    val_path = PROCESSED_DIR / "splits" / "val.json"
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_count = len(json.load(f).get("samples", []))
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_count = len(json.load(f).get("samples", []))
    failed = [row for row in target_audit if safe_int(row["current_failed_gt5"]) == 1]
    no_total = sum(row["bottleneck"] in {"candidate_generation_or_centerline", "lack_of_candidates"} for row in failed)
    selector_or_short = sum(row["bottleneck"] in {"selector_gate", "candidate_exists_not_chosen", "shortlist_or_anatomical_band"} for row in failed)
    severe_structural = sum(
        "caso severo raro" in row["classes"] or "multi-target" in row["classes"]
        for row in severe_rows
    )
    return [
        {"signal": "train_split_available", "value": train_count, "interpretation": "dataset disponivel para treino"},
        {"signal": "val_split_available", "value": val_count, "interpretation": "dataset disponivel para validacao"},
        {"signal": "current_train_used", "value": 2000, "interpretation": "script 04 limita treino atual"},
        {"signal": "current_val_used", "value": 400, "interpretation": "script 04 limita validacao atual"},
        {"signal": "failed_targets_without_good_total_candidate", "value": no_total, "interpretation": "sinal de geracao/centerline/segmentacao/treino"},
        {"signal": "failed_targets_with_candidate_selection_bottleneck", "value": selector_or_short, "interpretation": "sinal de pipeline selector/shortlist"},
        {"signal": "severe_images_structural_or_multitarget", "value": severe_structural, "interpretation": "sinal de severidade/distribuicao dificil"},
    ]


def summary_markdown(severe_rows, target_audit, severe_target_audit, bottleneck_rows, progress_rows, overlap_rows, ceiling_rows, train_rows):
    def count_where(rows, key, value=1):
        return sum(safe_int(row.get(key, 0)) == value for row in rows)

    failed_targets = [row for row in target_audit if safe_int(row["current_failed_gt5"]) == 1]
    severe_targets = [row for row in target_audit if safe_int(row["current_severe_gt8"]) == 1]
    bottleneck_counts = Counter(row["bottleneck"] for row in failed_targets)
    severe_bottleneck_counts = Counter(row["bottleneck"] for row in severe_targets)
    lines = [
        "# Auditoria 202 gargalo checkpoint201",
        "",
        "## Baseline atual",
        "",
    ]
    current = [row for row in progress_rows if row["checkpoint"] == "201"][0]
    for key in [
        "max_cobb_smape_article", "pt_smape_article", "mt_smape_article",
        "tll_smape_article", "agg3_smape_article", "mae3", "within_5",
        "failures_gt5", "severe_gt8",
    ]:
        lines.append(f"- {key}: {current[key]}")
    lines.extend([
        "",
        "## Severos atuais",
        "",
        f"- Imagens severas >8: {len(severe_rows)}.",
        f"- Multi-target: {count_where(severe_rows, 'multi_target')}.",
        f"- PT dominante: {count_where(severe_rows, 'pt_dominant')}.",
        f"- TL/L dominante: {count_where(severe_rows, 'tll_dominant')}.",
        f"- MT dominante: {count_where(severe_rows, 'mt_dominant')}.",
        f"- Centerline instavel/suspeita: {count_where(severe_rows, 'centerline_instavel')}.",
        f"- Mascara/segmentacao suspeita: {count_where(severe_rows, 'mascara_segmentacao_suspeita')}.",
        f"- Problema de banda anatomica/sweep/shortlist: {sum('problema de banda anatomica' in row['classes'] for row in severe_rows)}.",
        "",
        "## Gargalo quantitativo por target falhado",
        "",
    ])
    for bottleneck, count in bottleneck_counts.most_common():
        lines.append(f"- {bottleneck}: {count}")
    lines.extend(["", "## Gargalo quantitativo em targets severos", ""])
    for bottleneck, count in severe_bottleneck_counts.most_common():
        lines.append(f"- {bottleneck}: {count}")
    lines.extend(["", "## Oracle ceiling atual", ""])
    for row in ceiling_rows:
        lines.append(
            f"- {row['method']}: MAE3(abs) {row['mae3_abs_error_only']}, "
            f"<=5 {row['within_5_abs_error_only']}, falhas {row['failures_gt5_abs_error_only']}, "
            f"severos {row['severe_gt8_abs_error_only']}."
        )
    lines.extend(["", "## Sobreposicao de severos", ""])
    for row in overlap_rows:
        lines.append(
            f"- {row['comparison']}: mesmos {row['same_severe_count']}, "
            f"novos em 201 {row['new_in_201_vs_ref']}, corrigidos da referencia {row['fixed_from_ref']}."
        )
    lines.extend(["", "## Treino vs pipeline", ""])
    for row in train_rows:
        lines.append(f"- {row['signal']}: {row['value']} ({row['interpretation']}).")
    lines.extend([
        "",
        "## Diagnostico principal",
        "",
        "- O gargalo imediato ainda e pipeline: existe margem grande entre checkpoint201 e oracle_shortlist/oracle_total.",
        "- Dentro da pipeline, o maior bloqueio numerico e geracao/centerline/qualidade de candidatos, seguido de shortlist/banda anatomica e selector/gate.",
        "- Treino maior ja faz sentido em paralelo, porque a base usa 2000/12768 imagens e ha sinais de candidatos ausentes/instaveis, mas nao deve substituir a auditoria severe-first.",
        "",
        "## Proximo passo recomendado",
        "",
        "- O proximo ganho maior deve vir de uma validacao severe-first PT/TL-L sobre os 25 severos do checkpoint201.",
        "- Scripts recomendados: 203 para preparar/auditar shortlist severe-first, 204 para validar combinacoes PT/TL-L sem GT na inferencia, 205 para promover apenas se mantiver 0 regressões.",
    ])
    (OUTPUT_DIR / "audit202_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    severe_rows, severe_target_audit = severe_rows_201()
    progress_rows, overlap_rows = checkpoint_progress_rows()
    target_audit = current_target_rows()
    bottleneck_rows = bottleneck_summary(target_audit, severe_target_audit)
    ceiling_rows = oracle_ceiling_rows()
    train_rows = training_rows(severe_rows, target_audit, ceiling_rows)

    write_csv(OUTPUT_DIR / "audit202_severe25_rows.csv", severe_rows, union_fieldnames(severe_rows))
    write_csv(OUTPUT_DIR / "audit202_severe25_target_rows.csv", severe_target_audit, union_fieldnames(severe_target_audit))
    write_csv(OUTPUT_DIR / "audit202_checkpoint_progress.csv", progress_rows, union_fieldnames(progress_rows))
    write_csv(OUTPUT_DIR / "audit202_severe_overlap.csv", overlap_rows, union_fieldnames(overlap_rows))
    write_csv(OUTPUT_DIR / "audit202_target_bottleneck_rows.csv", target_audit, union_fieldnames(target_audit))
    write_csv(OUTPUT_DIR / "audit202_bottleneck_summary.csv", bottleneck_rows, union_fieldnames(bottleneck_rows))
    write_csv(OUTPUT_DIR / "audit202_oracle_ceiling.csv", ceiling_rows, union_fieldnames(ceiling_rows))
    write_csv(OUTPUT_DIR / "audit202_training_signals.csv", train_rows, union_fieldnames(train_rows))
    summary_markdown(severe_rows, target_audit, severe_target_audit, bottleneck_rows, progress_rows, overlap_rows, ceiling_rows, train_rows)

    print("\n===== AUDITORIA202 GARGALO CHECKPOINT201 =====")
    print(f"Severos auditados: {len(severe_rows)}")
    for row in ceiling_rows:
        print(
            f"{row['method']}: falhas {row['failures_gt5_abs_error_only']} | "
            f"severos {row['severe_gt8_abs_error_only']} | MAE3(abs) {row['mae3_abs_error_only']}"
        )
    print(f"Resumo: {OUTPUT_DIR / 'audit202_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
