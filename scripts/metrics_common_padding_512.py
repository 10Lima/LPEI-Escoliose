import math
from collections import defaultdict

import numpy as np


TARGETS = ["PT", "MT", "TL_L"]
EPS = 1e-12


def safe_float(value):
    if value is None or value == "":
        return np.nan
    return float(value)


def safe_int(value):
    if value is None or value == "":
        return 0
    return int(float(value))


def round_safe(value, digits=4):
    value = safe_float(value)
    return "" if not np.isfinite(value) else round(float(value), digits)


def smape_article(pred_values, gt_values):
    pred = np.asarray(pred_values, dtype=float)
    gt = np.asarray(gt_values, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(gt)
    if not np.any(valid):
        return np.nan
    numerator = float(np.sum(np.abs(pred[valid] - gt[valid])))
    denominator = float(np.sum(np.abs(pred[valid]) + np.abs(gt[valid])))
    return 0.0 if denominator <= EPS else 100.0 * numerator / denominator


def smape_old(pred_values, gt_values):
    value = smape_article(pred_values, gt_values)
    return value * 2.0 if np.isfinite(value) else value


def smape_target_article(pred, gt):
    pred = safe_float(pred)
    gt = safe_float(gt)
    if not np.isfinite(pred) or not np.isfinite(gt):
        return np.nan
    denominator = abs(pred) + abs(gt)
    return 0.0 if denominator <= EPS else 100.0 * abs(pred - gt) / denominator


def smape_target_old(pred, gt):
    value = smape_target_article(pred, gt)
    return value * 2.0 if np.isfinite(value) else value


def rows_to_arrays(target_rows, estimate_col="estimated", gt_col="gt"):
    by_file = defaultdict(dict)
    raw_ids = {}
    for row in target_rows:
        file_id = row["file_id"]
        target = row["target"]
        if target not in TARGETS:
            continue
        by_file[file_id][target] = row
        raw_ids[file_id] = row.get("raw_file_id", "")

    file_ids = sorted(file_id for file_id, values in by_file.items() if all(target in values for target in TARGETS))
    preds = []
    gts = []
    for file_id in file_ids:
        preds.append([safe_float(by_file[file_id][target][estimate_col]) for target in TARGETS])
        gts.append([safe_float(by_file[file_id][target][gt_col]) for target in TARGETS])
    return file_ids, raw_ids, np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)


def metric_summary_from_arrays(preds, gts, baseline_preds=None):
    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    if preds.shape != gts.shape or preds.ndim != 2 or preds.shape[1] != 3:
        raise ValueError(f"Shapes invalidos para metricas Cobb: preds={preds.shape}, gts={gts.shape}")

    abs_errors = np.abs(preds - gts)
    mae3_by_image = np.mean(abs_errors, axis=1)
    max_pred = np.max(preds, axis=1)
    max_gt = np.max(gts, axis=1)

    summary = {
        "max_cobb_smape_article": round_safe(smape_article(max_pred, max_gt)),
        "pt_smape_article": round_safe(smape_article(preds[:, 0], gts[:, 0])),
        "mt_smape_article": round_safe(smape_article(preds[:, 1], gts[:, 1])),
        "tll_smape_article": round_safe(smape_article(preds[:, 2], gts[:, 2])),
        "agg3_smape_article": round_safe(smape_article(preds.reshape(-1), gts.reshape(-1))),
        "max_cobb_smape_old": round_safe(smape_old(max_pred, max_gt)),
        "pt_smape_old": round_safe(smape_old(preds[:, 0], gts[:, 0])),
        "mt_smape_old": round_safe(smape_old(preds[:, 1], gts[:, 1])),
        "tll_smape_old": round_safe(smape_old(preds[:, 2], gts[:, 2])),
        "agg3_smape_old": round_safe(smape_old(preds.reshape(-1), gts.reshape(-1))),
        "mae3": round_safe(np.mean(mae3_by_image)),
        "rmse3": round_safe(math.sqrt(float(np.mean(abs_errors ** 2)))),
        "within_5": round_safe(np.mean(mae3_by_image <= 5.0) * 100.0),
        "within_10": round_safe(np.mean(mae3_by_image <= 10.0) * 100.0),
        "within_15": round_safe(np.mean(mae3_by_image <= 15.0) * 100.0),
        "failures_gt5": int(np.sum(mae3_by_image > 5.0)),
        "severe_gt8": int(np.sum(mae3_by_image > 8.0)),
        "case_count": int(preds.shape[0]),
    }

    if baseline_preds is not None:
        baseline_preds = np.asarray(baseline_preds, dtype=float)
        if baseline_preds.shape != preds.shape:
            raise ValueError(
                f"Shapes invalidos para regressoes: baseline={baseline_preds.shape}, preds={preds.shape}"
            )
        baseline_abs = np.abs(baseline_preds - gts)
        target_delta = abs_errors - baseline_abs
        image_delta = mae3_by_image - np.mean(baseline_abs, axis=1)
        summary.update({
            "regressions_target_gt5": int(np.sum(target_delta > 5.0)),
            "regressions_image_gt1": int(np.sum(image_delta > 1.0)),
            "regressions_image_gt3": int(np.sum(image_delta > 3.0)),
        })
    else:
        summary.update({
            "regressions_target_gt5": 0,
            "regressions_image_gt1": 0,
            "regressions_image_gt3": 0,
        })
    return summary


def print_required_metric_block(title, summary):
    print(f"\n{title}")
    print(f"  Max Cobb SMAPE_article: {summary['max_cobb_smape_article']}")
    print(f"  PT SMAPE_article: {summary['pt_smape_article']}")
    print(f"  MT SMAPE_article: {summary['mt_smape_article']}")
    print(f"  TL/L SMAPE_article: {summary['tll_smape_article']}")
    print(f"  Agg3 SMAPE_article: {summary['agg3_smape_article']}")
    print(f"  MAE3: {summary['mae3']}")
    print(f"  RMSE3: {summary['rmse3']}")
    print(f"  <=5: {summary['within_5']}%")
    print(f"  <=10: {summary['within_10']}%")
    print(f"  <=15: {summary['within_15']}%")
    print(f"  falhas >5: {summary['failures_gt5']}")
    print(f"  severos >8: {summary['severe_gt8']}")
    print(f"  regressoes target >5: {summary['regressions_target_gt5']}")
    print(f"  regressoes imagem >1: {summary['regressions_image_gt1']}")
    print(f"  regressoes imagem >3: {summary['regressions_image_gt3']}")
