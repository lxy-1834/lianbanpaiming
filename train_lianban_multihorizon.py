# -*- coding: utf-8 -*-
"""分阶段、多目标连板模型训练。

严格按时间切分：训练70%、验证15%、最终预测测试15%。
测试集不参与训练，只用于检验模型样本外表现。
"""
import json, math, os
from collections import defaultdict
import numpy as np

from train_lianban_model import fit_logistic, predict, board_num, load_data, safe

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "lianban_stage_dataset.jsonl")
POOLFILE = os.path.join(HERE, "ths_pool_history.json")
REPORT = os.path.join(HERE, "lianban_multihorizon_report.md")
MODELFILE = os.path.join(HERE, "lianban_multihorizon_model.json")
BASELINE = os.path.join(HERE, "lianban_multihorizon_model_baseline.json")


def k_features(row, kc):
    arr = kc.get(row["code"]) or []
    ds = f"{row['date'][:4]}-{row['date'][4:6]}-{row['date'][6:]}"
    i = next((j for j, p in enumerate(arr) if p[0] == ds), None)
    if i is None or i < 60:
        return {}
    try:
        close = safe(arr[i][2])
        closes = [safe(p[2]) for p in arr[:i + 1]]
        highs20 = [safe(p[3]) for p in arr[i - 20:i]]
        lows20 = [safe(p[4]) for p in arr[i - 20:i]]
        vols5 = [safe(p[5]) for p in arr[i - 5:i]]
        vols20 = [safe(p[5]) for p in arr[i - 20:i]]
        if close <= 0 or min(highs20 + lows20) <= 0:
            return {}
        ma5, ma10, ma20 = (np.mean(closes[-n:]) for n in (5, 10, 20))
        ma5_prev = np.mean(closes[-10:-5])
        ma10_prev = np.mean(closes[-20:-10])
        ma20_prev = np.mean(closes[-40:-20])
        return {
            "k_close_ma5": close / ma5 - 1, "k_close_ma10": close / ma10 - 1,
            "k_close_ma20": close / ma20 - 1,
            "k_ma5_slope": ma5 / ma5_prev - 1, "k_ma10_slope": ma10 / ma10_prev - 1,
            "k_ma20_slope": ma20 / ma20_prev - 1,
            "k_ret5": close / closes[-6] - 1, "k_ret10": close / closes[-11] - 1,
            "k_ret20": close / closes[-21] - 1, "k_ret60": close / closes[-61] - 1,
            "k_break10": close / max(safe(p[2]) for p in arr[i - 10:i]) - 1,
            "k_break20": close / max(safe(p[2]) for p in arr[i - 20:i]) - 1,
            "k_amp20": (max(highs20) - min(lows20)) / min(lows20),
            "k_vol_ratio5": safe(arr[i][5]) / (np.mean(vols5) or 1),
            "k_vol_ratio20": safe(arr[i][5]) / (np.mean(vols20) or 1),
            "k_vol5_vs20": (np.mean(vols5) / (np.mean(vols20) or 1)) - 1,
        }
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return {}


def metrics(y, p):
    n = len(y)
    if not n:
        return {"n": 0, "base": None, "top10": None, "lift": None}
    k = max(1, math.ceil(n * .1))
    top = np.argsort(-p)[:k]
    base = float(y.mean())
    hit = float(y[top].mean())
    return {"n": n, "base": base, "top10": hit, "lift": hit / base if base else None}


def main():
    rows = [json.loads(x) for x in open(DATASET, encoding="utf-8")]
    pool, kc, _ = load_data()
    dates = sorted(d for d, v in pool.items() if v)
    didx = {d: i for i, d in enumerate(dates)}
    # 构造每条样本未来20个交易日内的最高板位，不能把它用于特征。
    for r in rows:
        i = didx.get(r["date"])
        future = []
        # 必须有完整20个后续交易日，否则剔除，避免末端标签被截断。
        if i is None or i + 20 >= len(dates):
            r["_complete"] = False
            continue
        for d in dates[i + 1:i + 21]:
            future.extend(board_num(x) for x in pool[d] if x.get("code") == r["code"])
        r["future_max"] = max(future or [int(r.get("stage") or 1)])
        r["_complete"] = True
    rows = [r for r in rows if r.get("_complete")]
    for r in rows:
        r.update(k_features(r, kc))

    feature_names = [
        "stage", "turnover_rate", "currency_value", "order_amount", "order_volume",
        "limit_up_suc_rate", "is_again_limit", "change_rate", "tag_count", "theme_heat",
        "theme_max_board", "theme_sum_3d", "theme_active_days_3d", "theme_max_board_3d",
        "theme_sum_5d", "theme_active_days_5d", "theme_max_board_5d",
        "theme_sum_10d", "theme_active_days_10d", "theme_max_board_10d",
        "k_close_ma5", "k_close_ma10", "k_close_ma20", "k_ma5_slope", "k_ma10_slope",
        "k_ma20_slope", "k_ret5", "k_ret10", "k_ret20", "k_ret60", "k_break10",
        "k_break20", "k_amp20", "k_vol_ratio5", "k_vol_ratio20", "k_vol5_vs20",
    ]
    # 金额类压缩，避免流通市值量级主导梯度。
    def val(r, n):
        x = r.get(n, 0)
        try: x = float(x or 0)
        except (TypeError, ValueError): x = 0.0
        if n in ("currency_value", "order_amount", "order_volume"):
            return math.log1p(max(x, 0))
        return x if math.isfinite(x) else 0.0
    x = np.array([[val(r, n) for n in feature_names] for r in rows], dtype=float)
    dates_rows = sorted({r["date"] for r in rows})
    c1 = dates_rows[int(len(dates_rows) * .70)]
    c2 = dates_rows[int(len(dates_rows) * .85)]
    split = lambda r: "train" if r["date"] <= c1 else ("valid" if r["date"] <= c2 else "test")
    result = {"feature_names": feature_names, "split": {"train_end": c1, "valid_end": c2}, "models": {}}
    report = ["# 分阶段多目标连板模型（日K+量能版）", "", f"样本：{len(rows)}；特征：{len(feature_names)}（含日K和量能）；预测窗口：未来20个交易日。", "", f"时间切分：训练 ≤ {c1}；验证 {c1}~{c2}；最终测试 > {c2}。", "", "测试集完全未参与训练。", ""]
    baseline = json.load(open(BASELINE, encoding="utf-8")) if os.path.exists(BASELINE) else {}
    comparison = []

    for group in ["1", "2", "3", "4", "5", "6", "7+"]:
        idx = np.array([i for i, r in enumerate(rows) if r["stage_group"] == group])
        if len(idx) < 80:
            continue
        stage = 7 if group == "7+" else int(group)
        result["models"].setdefault(group, {})
        report.append(f"## 当前 {group} 板")
        report.append("|目标|训练正率|验证前10%|测试前10%|测试基准|测试提升|")
        report.append("|---|---:|---:|---:|---:|---:|")
        for ahead in (1, 2, 3, 4, 6):
            target = stage + ahead
            y = np.array([1.0 if rows[i]["future_max"] >= target else 0.0 for i in idx])
            tr = np.array([rows[i]["date"] <= c1 for i in idx])
            va = np.array([c1 < rows[i]["date"] <= c2 for i in idx])
            te = np.array([rows[i]["date"] > c2 for i in idx])
            if tr.sum() < 40 or y[tr].sum() < 3 or (y[tr] == 0).sum() < 3:
                continue
            model = fit_logistic(x[idx][tr], y[tr])
            pv = predict(model, x[idx])
            mm = {k: metrics(y[mask], pv[mask]) for k, mask in (("train", tr), ("valid", va), ("test", te))}
            result["models"][group][f"reach_{target}"] = {
                "ahead": ahead, "metrics": mm,
                "w": model[0].tolist(), "mu": model[1].tolist(), "sd": model[2].tolist(),
            }
            t = mm["test"]
            if t["n"]:
                report.append(f"|达到{target}板|{mm['train']['base']:.1%}|{mm['valid']['top10']:.1%}|{t['top10']:.1%}|{t['base']:.1%}|{t['lift']:.2f}x|")
                old = baseline.get("models", {}).get(group, {}).get(f"reach_{target}", {}).get("metrics", {}).get("test", {})
                if old.get("top10") is not None:
                    comparison.append((group, target, old.get("top10", 0), t["top10"], old.get("lift", 0), t["lift"]))
        report.append("")
    report.extend(["## 与原模型同测试集对比", "", "|当前板|目标|原模型前10%|日K量能版前10%|原提升|新提升|", "|---|---|---:|---:|---:|---:|"])
    for group, target, old_hit, new_hit, old_lift, new_lift in comparison:
        report.append(f"|{group}|{target}板|{old_hit:.1%}|{new_hit:.1%}|{old_lift:.2f}x|{new_lift:.2f}x|")
    json.dump(result, open(MODELFILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(REPORT, "w", encoding="utf-8").write("\n".join(report) + "\n")
    print(f"完成：训练/验证/测试已分离，样本 {len(rows)} 条，新增日K/量能特征16个")
    print(f"训练截止：{c1} | 验证截止：{c2} | 测试集：{sum(r['date'] > c2 for r in rows)} 条")
    print(f"报告：{REPORT}")
    print(f"模型：{MODELFILE}")


if __name__ == "__main__":
    main()
