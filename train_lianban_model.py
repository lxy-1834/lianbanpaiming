# -*- coding: utf-8 -*-
"""首板连板潜力模型（仅使用首板收盘时可见信息）。

标签：首板后 20 个交易日内是否达到 2/3/5/8 板。
输出：work/lianban_model_report.md、work/lianban_model_result.json
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "ths_pool_history.json")
KFILES = ("ths_daily_cache.json", "all_daily_cache.json")
REPORT = os.path.join(HERE, "lianban_model_report.md")
RESULT = os.path.join(HERE, "lianban_model_result.json")


def board_num(item):
    encoded = item.get("high_days_value")
    try:
        if encoded is not None:
            value = int(encoded)
            if value > 65536:
                return value & 65535
    except (TypeError, ValueError):
        pass
    m = re.match(r"(\d+)", str(item.get("high_days") or ""))
    return int(m.group(1)) if m else 0


def is_first_board(item):
    return board_num(item) == 1


def tags(item):
    return [x.strip() for x in str(item.get("reason_type") or "").split("+") if x.strip()]


def safe(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def load_data():
    with open(POOL, encoding="utf-8") as f:
        pool = json.load(f)
    kc = {}
    for fn in KFILES:
        fp = os.path.join(HERE, fn)
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for code, arr in data.items():
            if arr and (code not in kc or len(arr) > len(kc[code])):
                kc[code] = arr
    dates = sorted(d for d, rows in pool.items() if rows)
    return pool, kc, dates


def build_rows(pool, kc, dates):
    didx = {d: i for i, d in enumerate(dates)}
    daily_tags = {}
    daily_max_board = {}
    for d in dates:
        cnt = Counter()
        mx = defaultdict(int)
        for item in pool[d]:
            b = board_num(item)
            for t in tags(item):
                cnt[t] += 1
                mx[t] = max(mx[t], b)
        daily_tags[d] = cnt
        daily_max_board[d] = mx

    rows = []
    for d in dates[:-20]:
        di = didx[d]
        prior = {n: dates[max(0, di - n):di] for n in (3, 5, 10)}
        for item in pool[d]:
            if item.get("market_type") != "HS" or not is_first_board(item):
                continue
            code = item.get("code")
            arr = kc.get(code) or []
            ds = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            ki = next((i for i, p in enumerate(arr) if p[0] == ds), None)
            if ki is None or ki < 60:
                continue
            try:
                close = safe(arr[ki][2])
                prev5 = arr[ki - 5:ki]
                prev10 = arr[ki - 10:ki]
                prev20 = arr[ki - 20:ki]
                prev60 = arr[ki - 60:ki]
                closes5 = [safe(p[2]) for p in prev5]
                closes10 = [safe(p[2]) for p in prev10]
                closes20 = [safe(p[2]) for p in prev20]
                closes60 = [safe(p[2]) for p in prev60]
                highs10 = [safe(p[3]) for p in prev10]
                highs20 = [safe(p[3]) for p in prev20]
                lows20 = [safe(p[4]) for p in prev20]
                lows60 = [safe(p[4]) for p in prev60]
                highs60 = [safe(p[3]) for p in prev60]
                if min(closes20 + lows20 + lows60) <= 0:
                    continue
            except (IndexError, TypeError):
                continue

            ts = tags(item)
            heat = max((daily_tags[d].get(t, 0) for t in ts), default=0)
            theme_features = {}
            for n in (3, 5, 10):
                counts = [daily_tags[dd].get(t, 0) for dd in prior[n] for t in ts]
                theme_features[f"theme_sum{n}"] = sum(counts)
                theme_features[f"theme_days{n}"] = sum(v > 0 for v in counts)
                theme_features[f"theme_maxboard{n}"] = max(
                    [daily_max_board[dd].get(t, 0) for dd in prior[n] for t in ts] or [0]
                )
            future = []
            for dd in dates[di + 1:di + 21]:
                future.extend(board_num(x) for x in pool[dd] if x.get("code") == code)
            if not future:
                continue
            max_future = max(future)
            turnover = safe(item.get("turnover_rate"))
            currency = safe(item.get("currency_value"))
            order_amount = safe(item.get("order_amount"))
            amount = currency * turnover / 100 if currency and turnover else 0
            try:
                fbt = int(item.get("first_limit_up_time"))
                import datetime
                fbmin = datetime.datetime.fromtimestamp(fbt).hour * 60 + datetime.datetime.fromtimestamp(fbt).minute
            except (TypeError, ValueError, OverflowError, OSError):
                fbmin = 0
            feat = {
                "turnover": turnover,
                "amount_log": math.log1p(amount),
                "float_mv_log": math.log1p(currency),
                "seal_ratio": order_amount / amount * 100 if amount else 0,
                "order_amount_log": math.log1p(order_amount),
                "fbmin": fbmin,
                "is_one_word": 1 if "一字" in str(item.get("limit_up_type") or "") else 0,
                "open_num": safe(item.get("open_num")),
                "change_rate": safe(item.get("change_rate")),
                "tag_count": len(ts),
                "theme_heat": heat,
                "close_vs_ma5": close / np.mean(closes5) - 1,
                "close_vs_ma10": close / np.mean(closes10) - 1,
                "close_vs_ma20": close / np.mean(closes20) - 1,
                "ret5": close / closes5[0] - 1,
                "ret10": close / closes10[0] - 1,
                "ret20": close / closes20[0] - 1,
                "ret60": close / closes60[0] - 1,
                "break10": close / max(highs10) - 1,
                "break20": close / max(highs20) - 1,
                "pos60": (close - min(lows60)) / (max(highs60) - min(lows60) or close),
                "amp20": (max(highs20) - min(lows20)) / min(lows20),
            }
            feat.update(theme_features)
            rows.append({"date": d, "code": code, "name": item.get("name"), "max_future_board": max_future, "features": feat})
    return rows


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(x, y):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-9] = 1
    z = (x - mu) / sd
    w = np.zeros(z.shape[1] + 1)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, min(neg / pos, 12.0), 1.0)
    for _ in range(1800):
        p = sigmoid(w[0] + z @ w[1:])
        err = (p - y) * weights
        grad0 = err.mean()
        grad = (z.T @ err) / len(y) + 0.01 * w[1:]
        w[0] -= 0.08 * grad0
        w[1:] -= 0.08 * grad
    return w, mu, sd


def predict(model, x):
    w, mu, sd = model
    return sigmoid(w[0] + ((x - mu) / sd) @ w[1:])


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    neg = y == 0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())) if pos.any() and neg.any() else None


def metrics(y, p):
    n = len(y)
    topn = max(1, math.ceil(n * 0.1))
    top = np.argsort(-p)[:topn]
    base = float(y.mean()) if n else 0
    hit = float(y[top].mean()) if n else 0
    return {"n": n, "base": base, "top10": hit, "lift": hit / base if base else None, "auc": auc(y, p)}


def main():
    pool, kc, dates = load_data()
    rows = build_rows(pool, kc, dates)
    unique = sorted({r["date"] for r in rows})
    cut1 = unique[int(len(unique) * 0.70)]
    cut2 = unique[int(len(unique) * 0.85)]
    names = list(rows[0]["features"])
    x = np.array([[r["features"][n] for n in names] for r in rows], dtype=float)
    x[~np.isfinite(x)] = 0
    result = {"sample_count": len(rows), "feature_names": names, "split": {"train_end": cut1, "valid_end": cut2}, "targets": {}}
    report = ["# 首板连板潜力模型第一版", "", f"样本：{len(rows)} 条；特征：{len(names)} 个；数据只使用首板当天字段。", "", f"时间切分：训练 ≤ {cut1}，验证 {cut1}~{cut2}，样本外测试 > {cut2}。", ""]
    for target in (2, 3, 5, 8):
        y = np.array([1 if r["max_future_board"] >= target else 0 for r in rows], dtype=float)
        tr = np.array([r["date"] <= cut1 for r in rows])
        va = np.array([cut1 < r["date"] <= cut2 for r in rows])
        te = np.array([r["date"] > cut2 for r in rows])
        model = fit_logistic(x[tr], y[tr])
        pp = predict(model, x)
        met = {"train": metrics(y[tr], pp[tr]), "valid": metrics(y[va], pp[va]), "test": metrics(y[te], pp[te])}
        coef = sorted(zip(names, model[0][1:]), key=lambda z: abs(z[1]), reverse=True)[:10]
        result["targets"][f"reach_{target}"] = {"metrics": met, "top_coefficients": coef}
        report.append(f"## 达到 {target} 板")
        report.append(f"总体基准：{y.mean():.1%}（正样本 {int(y.sum())}/{len(y)}）")
        report.append("|区间|样本|基准|前10%命中率|提升|AUC|")
        report.append("|---|---:|---:|---:|---:|---:|")
        for part in ("train", "valid", "test"):
            m = met[part]
            report.append(f"|{part}|{m['n']}|{m['base']:.1%}|{m['top10']:.1%}|{m['lift']:.2f}x|{m['auc']:.3f}|" if m['auc'] is not None else f"|{part}|{m['n']}|{m['base']:.1%}|{m['top10']:.1%}|-|无|")
        report.append("主要系数：" + "、".join(f"{n}({c:+.2f})" for n, c in coef))
        report.append("")
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"完成：样本 {len(rows)} 条，特征 {len(names)} 个")
    print(f"报告：{REPORT}")
    for target, out in result["targets"].items():
        m = out["metrics"]["test"]
        print(f"{target}: 测试集 n={m['n']} 前10%命中={m['top10']:.1%} 基准={m['base']:.1%} 提升={m['lift']:.2f}x AUC={m['auc']:.3f}" if m['auc'] is not None else f"{target}: 测试集样本不足")


if __name__ == "__main__":
    main()
