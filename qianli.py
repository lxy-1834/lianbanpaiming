# -*- coding: utf-8 -*-
"""千里：首板连板潜力筛选器。

直接运行：D:\\python\\python.exe qianli.py
程序会读取历史数据训练模型，并获取当天同花顺涨停池后输出首板候选。
"""
import datetime
import html
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lianban_model import (
    load_data, safe, tags, board_num,
)
from train_lianban_multihorizon import k_features

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_STORE = os.path.join(HERE, "ths_pool_history.json")
PREDICTION_LOG = os.path.join(HERE, "qianli_prediction_log.jsonl")
UA = {"User-Agent": "Mozilla/5.0"}
FIELDS = "199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004"


def today_pool():
    r = requests.get(
        "https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool",
        params={"page": 1, "limit": 200, "field": FIELDS,
                "filter": "HS,GEM2STAR", "order_field": "330324",
                "order_type": 0},
        timeout=25, headers=UA,
    )
    return (r.json().get("data") or {}).get("info") or []


def save_daily_snapshot(date, live):
    """保存同花顺当日快照，并记录模型当日预测，避免重复追加。"""
    pool = {}
    if os.path.exists(POOL_STORE):
        try:
            with open(POOL_STORE, encoding="utf-8") as f:
                pool = json.load(f)
        except (OSError, ValueError):
            pool = {}
    pool[date] = live
    with open(POOL_STORE, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False)


def feature(item, d, dates, pool, kc):
    arr = kc.get(item.get("code")) or []
    ds = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    ki = next((i for i, p in enumerate(arr) if p[0] == ds), None)
    if ki is None or ki < 60:
        return None
    prev5, prev10 = arr[ki - 5:ki], arr[ki - 10:ki]
    prev20, prev60 = arr[ki - 20:ki], arr[ki - 60:ki]
    close = safe(arr[ki][2])
    c5 = [safe(p[2]) for p in prev5]
    c10 = [safe(p[2]) for p in prev10]
    c20 = [safe(p[2]) for p in prev20]
    c60 = [safe(p[2]) for p in prev60]
    h10 = [safe(p[3]) for p in prev10]
    h20 = [safe(p[3]) for p in prev20]
    l20 = [safe(p[4]) for p in prev20]
    l60 = [safe(p[4]) for p in prev60]
    h60 = [safe(p[3]) for p in prev60]
    if min(c20 + l20 + l60) <= 0:
        return None
    cnt = Counter()
    mx = defaultdict(int)
    for x in pool.get(d, []):
        b = board_num(x)
        for t in tags(x):
            cnt[t] += 1
            mx[t] = max(mx[t], b)
    ts = tags(item)
    heat = max((cnt.get(t, 0) for t in ts), default=0)
    di = dates.index(d)
    prior = {n: dates[max(0, di - n):di] for n in (3, 5, 10)}
    daily_cnt, daily_mx = {}, {}
    for dd in dates:
        dc, dm = Counter(), defaultdict(int)
        for x in pool.get(dd, []):
            b = board_num(x)
            for t in tags(x):
                dc[t] += 1
                dm[t] = max(dm[t], b)
        daily_cnt[dd], daily_mx[dd] = dc, dm
    turnover = safe(item.get("turnover_rate"))
    currency = safe(item.get("currency_value"))
    order_amount = safe(item.get("order_amount"))
    amount = currency * turnover / 100 if currency and turnover else 0
    try:
        stamp = datetime.datetime.fromtimestamp(int(item.get("first_limit_up_time")))
        fbmin = stamp.hour * 60 + stamp.minute
    except (TypeError, ValueError, OverflowError, OSError):
        fbmin = 0
    out = {
        "turnover": turnover, "amount_log": math.log1p(amount),
        "float_mv_log": math.log1p(currency),
        "seal_ratio": order_amount / amount * 100 if amount else 0,
        "order_amount_log": math.log1p(order_amount), "fbmin": fbmin,
        "is_one_word": 1 if "一字" in str(item.get("limit_up_type") or "") else 0,
        "open_num": safe(item.get("open_num")),
        "change_rate": safe(item.get("change_rate")), "tag_count": len(ts),
        "theme_heat": heat, "close_vs_ma5": close / np.mean(c5) - 1,
        "close_vs_ma10": close / np.mean(c10) - 1,
        "close_vs_ma20": close / np.mean(c20) - 1,
        "ret5": close / c5[0] - 1, "ret10": close / c10[0] - 1,
        "ret20": close / c20[0] - 1, "ret60": close / c60[0] - 1,
        "break10": close / max(h10) - 1, "break20": close / max(h20) - 1,
        "pos60": (close - min(l60)) / (max(h60) - min(l60) or close),
        "amp20": (max(h20) - min(l20)) / min(l20),
    }
    for n in (3, 5, 10):
        vals = [daily_cnt[dd].get(t, 0) for dd in prior[n] for t in ts]
        out[f"theme_sum{n}"] = sum(vals)
        out[f"theme_days{n}"] = sum(v > 0 for v in vals)
        out[f"theme_maxboard{n}"] = max([daily_mx[dd].get(t, 0) for dd in prior[n] for t in ts] or [0])
    return out


def stage_feature(item, d, dates, pool):
    """与 train_lianban_multihorizon.py 完全一致的20个实盘特征。"""
    ts = tags(item)
    cnt = Counter()
    mx = defaultdict(int)
    for x in pool.get(d, []):
        b = board_num(x)
        for t in tags(x):
            cnt[t] += 1
            mx[t] = max(mx[t], b)
    di = dates.index(d)
    out = {
        "stage": board_num(item),
        "turnover_rate": safe(item.get("turnover_rate")),
        "currency_value": safe(item.get("currency_value")),
        "order_amount": safe(item.get("order_amount")),
        "order_volume": safe(item.get("order_volume")),
        "limit_up_suc_rate": safe(item.get("limit_up_suc_rate")),
        "is_again_limit": safe(item.get("is_again_limit")),
        "change_rate": safe(item.get("change_rate")),
        "tag_count": len(ts),
        "theme_heat": max((cnt.get(t, 0) for t in ts), default=0),
        "theme_max_board": max((mx.get(t, 0) for t in ts), default=0),
    }
    for n in (3, 5, 10):
        counts, maxboards = [], []
        for pd in dates[max(0, di - n):di]:
            pc, pm = Counter(), defaultdict(int)
            for x in pool.get(pd, []):
                b = board_num(x)
                for t in tags(x):
                    pc[t] += 1
                    pm[t] = max(pm[t], b)
            counts.extend(pc[t] for t in ts)
            maxboards.extend(pm[t] for t in ts)
        out[f"theme_sum_{n}d"] = sum(counts)
        out[f"theme_active_days_{n}d"] = sum(v > 0 for v in counts)
        out[f"theme_max_board_{n}d"] = max(maxboards or [0])
    return out


def model_value(v, name):
    v = safe(v)
    if name in ("currency_value", "order_amount", "order_volume"):
        return math.log1p(max(v, 0))
    return v


def save_html_report(path, date, candidates):
    """生成单文件可视化报告；排序与控制台一致，综合分从高到低。"""
    rows = []
    for i, r in enumerate(candidates, 1):
        p = r["prob"]
        best_h, best_p = max(p.items(), key=lambda z: z[1])
        cells = [
            i, r.get("name") or "", r.get("code") or "", f"{r['stage']}板",
            f"{r['theme_max_board']}板", f"{r['score']:.1%}",
            f"{r['expected_max_board']:.1f}板", f"{best_h}板 ({best_p:.0%})",
            f"{r['features'].get('turnover_rate', 0):.1f}%", r.get("reason") or "",
        ]
        cls = "top" if i <= 3 else ""
        rows.append(
            f'<tr class="{cls}">' + "".join(
                f"<td>{html.escape(str(v))}</td>" for v in cells
            ) + "</tr>"
        )
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_name = html.escape(str(candidates[0]["name"])) if candidates else "—"
    top_score = f"{candidates[0]['score']:.1%}" if candidates else "—"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>千里筛选结果 {html.escape(date)}</title>
<style>
:root {{ color-scheme: light; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
body {{ margin:0; background:#f3f6fb; color:#172033; }}
.wrap {{ max-width:1400px; margin:0 auto; padding:28px 18px 48px; }}
.hero {{ background:linear-gradient(135deg,#172554,#2563eb); color:white; border-radius:20px; padding:26px 30px; box-shadow:0 12px 30px #1e3a8a33; }}
h1 {{ margin:0 0 8px; font-size:28px; }} .sub {{ opacity:.85; }}
.stats {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:20px; }}
.stat {{ background:#ffffff22; border:1px solid #ffffff35; border-radius:12px; padding:12px 16px; min-width:120px; }}
.stat b {{ display:block; font-size:21px; margin-bottom:3px; }}
.panel {{ margin-top:20px; background:white; border-radius:16px; padding:16px; box-shadow:0 5px 18px #17203312; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:1050px; }}
th {{ background:#eef4ff; color:#334155; position:sticky; top:0; z-index:1; }}
th,td {{ padding:12px 10px; border-bottom:1px solid #e8edf5; text-align:left; white-space:nowrap; }}
tr:hover {{ background:#f8fbff; }} tr.top td {{ font-weight:600; }}
tr.top td:nth-child(2) {{ color:#d97706; }} td:nth-child(6) {{ color:#2563eb; font-weight:700; }}
.note {{ color:#64748b; font-size:13px; margin-top:14px; }}
</style></head><body><main class="wrap">
<section class="hero"><h1>千里 · 连板潜力筛选</h1>
<div class="sub">{html.escape(date)}　|　生成时间：{generated}　|　按综合分从高到低排列</div>
<div class="stats"><div class="stat"><b>{len(candidates)}</b>候选股票</div>
<div class="stat"><b>{top_name}</b>综合分最高</div>
<div class="stat"><b>{top_score}</b>最高综合分</div></div></section>
<section class="panel"><table><thead><tr>
<th>排名</th><th>股票</th><th>代码</th><th>当前板</th><th>题材最高板</th><th>综合分</th>
<th>预计高度</th><th>最强目标</th><th>换手率</th><th>题材/原因</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="note">提示：这是模型排序结果，不是买入保证；重点结合题材持续性、封板质量和次日开盘情况人工复核。</div>
</section></main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(document)


def main():
    pool, kc, dates = load_data()
    model_path = os.path.join(HERE, "lianban_rolling_model.json")
    if not os.path.exists(model_path):
        print("缺少多目标模型文件，请先运行 train_lianban_multihorizon.py。")
        return
    with open(model_path, encoding="utf-8") as f:
        trained = json.load(f)
    feature_names = trained["feature_names"]
    models = trained["models"]
    live = today_pool()
    if not live:
        print("今天没有获取到同花顺涨停池数据，请稍后重试。")
        return
    today = datetime.datetime.now().strftime("%Y%m%d")
    save_daily_snapshot(today, live)
    pool[today] = live
    if today not in dates:
        dates.append(today)
        dates.sort()
    candidates = []
    theme_max_board = defaultdict(int)
    for x in live:
        b = board_num(x)
        for t in tags(x):
            theme_max_board[t] = max(theme_max_board[t], b)
    for item in live:
        if item.get("market_type") != "HS" or board_num(item) < 1:
            continue
        group = "7+" if board_num(item) >= 7 else str(board_num(item))
        if group not in models:
            continue
        f = stage_feature(item, today, dates, pool)
        f.update(k_features({"code": item.get("code"), "date": today}, kc))
        v = np.array([[model_value(f.get(n), n) for n in feature_names]], dtype=float)
        probs = {}
        for key, m in models[group].items():
            w, mu, sd = np.array(m["w"]), np.array(m["mu"]), np.array(m["sd"])
            cols = m.get("feature_indices") or list(range(len(feature_names)))
            z = (v[0][cols] - mu) / np.where(sd == 0, 1, sd)
            target_height = int(key.split("_")[-1])
            # 7+模型只对当前板位之后的高度有意义；例如14板不再显示8~13板概率。
            if target_height > board_num(item):
                probs[target_height] = float(1 / (1 + np.exp(-np.clip(w[0] + z @ w[1:], -30, 30))))
        if not probs:
            continue
        # 远期目标权重更高，但仍保留晋级近目标的影响。
        score = sum(p * (0.5 + 0.1 * max(0, h - board_num(item))) for h, p in probs.items()) / max(sum(0.5 + 0.1 * max(0, h - board_num(item)) for h in probs), 1)
        expected = board_num(item) + sum(probs.values())
        item_theme_max = max((theme_max_board.get(t, 0) for t in tags(item)), default=board_num(item))
        candidates.append({"code": item.get("code"), "name": item.get("name"),
                           "stage": board_num(item), "stage_group": group,
                           "theme_max_board": item_theme_max,
                           "reason": item.get("reason_type"), "score": score,
                           "expected_max_board": expected, "prob": probs, "features": f})
    candidates.sort(key=lambda z: z["score"], reverse=True)
    out_path = os.path.join(HERE, f"qianli_{today}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": today, "candidates": candidates}, ensure_ascii=False) + "\n")
    # 页面固定写入同一个文件，避免每天累积 HTML 文件；运行时覆盖为最新结果。
    html_path = os.path.join(HERE, "qianli_latest.html")
    save_html_report(html_path, today, candidates)
    print("\n" + "=" * 92)
    print(f"千里多目标连板潜力筛选  {today}  |  候选 {len(candidates)} 只")
    print("=" * 92)
    print("排名  股票          个股当前板  题材最高板  综合分   预计高度  最强目标    换手   题材")
    print("-" * 104)
    for i, r in enumerate(candidates[:15], 1):
        p = r["prob"]
        best_h, best_p = max(p.items(), key=lambda z: z[1])
        best = f"{best_h}板({best_p:.0%})"
        print(f"{i:>2}    {str(r['name'])[:8]:<8} {r['code']}    {r['stage']:>2}板       {r['theme_max_board']:>2}板      {r['score']:.1%}    {r['expected_max_board']:.1f}板    "
              f"{best:<10} "
              f"{r['features']['turnover_rate']:.1f}%  {str(r['reason'])[:30]}")
    print("-" * 92)
    print(f"结果已保存：{out_path}")
    print(f"可视化页面：{html_path}")
    print("说明：这是多目标连板潜力排序，不是买入保证；优先查看前5名的题材和盘口质量。")


if __name__ == "__main__":
    main()
