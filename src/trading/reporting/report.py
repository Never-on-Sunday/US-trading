"""Builds a self-contained HTML report for one experiment run from artifacts/runs/<name>/."""

import html
import json
from pathlib import Path

import polars as pl

from trading.shared.storage import ARTIFACTS_DIR

SERIES = [  # (key, label, file stem, palette slot) — fixed order so colour follows the entity
    ("bot_static", "Bot — trained once", "equity_static_base", 1),
    ("bot_monthly", "Bot — retrained monthly", "equity_monthly_base", 2),
    ("SPY", "SPY (S&P 500)", "equity_bench_SPY", 3),
    ("QQQ", "QQQ (Nasdaq-100)", "equity_bench_QQQ", 4),
    ("SMH", "SMH (semiconductors)", "equity_bench_SMH", 5),
    ("basket", "Buy & hold all 50 stocks", "equity_bench_basket_equal_weight", 6),
]


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _row(cells: list[str], tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def build_report(name: str = "baseline_v1") -> Path:
    run = ARTIFACTS_DIR / "runs" / name
    res = json.loads((run / "result.json").read_text())
    cash = float(res["config"]["starting_cash"])
    hold, bench, val = res["holdout"], res["holdout_benchmarks"], res["validation"]
    sel = val["selected"]

    series = []
    for key, label, stem, slot in SERIES:
        f = run / f"{stem}.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f).sort("session_date")
        series.append({"key": key, "label": label, "slot": slot,
                       "dates": [str(x) for x in d["session_date"]], "values": [round(v, 2) for v in d["equity"]]})

    modes = [m for m in ("static", "monthly") if m in hold]
    main_mode = max(modes, key=lambda m: hold[m]["base"]["total_return"])
    bot = hold[main_mode]["base"]
    best_bench_key = max(bench, key=lambda k: bench[k]["total_return"])
    best_bench = bench[best_bench_key]
    bench_names = {"SPY": "SPY", "QQQ": "QQQ", "SMH": "SMH", "basket_equal_weight": "Buy & hold all 50 stocks"}
    mode_names = {"static": "Bot — trained once", "monthly": "Bot — retrained monthly"}
    beat = [bench_names[k] for k, v in bench.items() if bot["total_return"] > v["total_return"]]

    if bot["total_return"] > best_bench["total_return"]:
        verdict = "The bot beat every benchmark in the test period."
    elif bot["total_return"] > 0 and beat:
        verdict = f"The bot made money but only beat {', '.join(beat)}."
    elif bot["total_return"] > 0:
        verdict = "The bot made a small profit but every buy-and-hold benchmark made more."
    else:
        verdict = "The bot lost money after fees, while buy-and-hold benchmarks did better."

    perf_rows = []
    for m in modes:
        b = hold[m]["base"]
        perf_rows.append([mode_names[m], _usd(b["final_equity"]), _pct(b["total_return"]), f"{b['sharpe']:.2f}",
                          _pct(b["max_drawdown"]), f"{b['trades']:,}", _usd(b["fees"])])
    for k, v in sorted(bench.items(), key=lambda kv: -kv[1]["total_return"]):
        perf_rows.append([bench_names[k], _usd(v["final_equity"]), _pct(v["total_return"]), f"{v['sharpe']:.2f}",
                          _pct(v["max_drawdown"]), "1" if k != "basket_equal_weight" else str(v.get("n_stocks", "")), "—"])

    variant_names = {"no_costs": "No costs at all (not achievable)", "base": "Binance fees + spread (realistic)",
                     "costs_2x": "Double costs (stress test)", "t1_settlement": "Sale money usable next day only"}
    cost_rows = [[variant_names[v]] + [f"{_usd(hold[m][v]['final_equity'])} ({_pct(hold[m][v]['total_return'])})" for m in modes]
                 for v in ("no_costs", "base", "costs_2x", "t1_settlement")]

    val_rows = []
    for hz in val["by_horizon"]:
        ok = [r for r in hz["grid"] if r["trades"] >= res["config"]["min_validation_trades"]]
        top = max(ok, key=lambda r: r["total_return"]) if ok else None
        d = hz["diagnostics"]
        val_rows.append([
            f"{hz['horizon'] * 5} min", f"{d['ic_mean']:.4f}", f"{d['top0.99']['mean_fwd'] * 1e4:+.1f} bp",
            f"{sum(r['total_return'] > 0 for r in ok)} of {len(ok)}",
            (f"{_pct(top['total_return'])} ({top['trades']} trades)" if top else "too few trades"),
        ])
    val_bench = ", ".join(f"{bench_names[k]} {_pct(v['total_return'])}" for k, v in val["benchmarks"].items())

    trades_html = ""
    tf = run / f"trades_{main_mode}.parquet"
    if tf.exists() and (tr := pl.read_parquet(tf)).height:
        by = (tr.with_columns((pl.col("proceeds") - pl.col("cost")).alias("pnl"))
              .group_by("symbol").agg(pl.len().alias("n"), pl.col("pnl").sum(), (pl.col("pnl") > 0).mean().alias("win"))
              .sort("pnl", descending=True))
        pick = pl.concat([by.head(5), by.tail(5)]).unique("symbol", maintain_order=True)
        trades_html = ("<table><thead>" + _row(["Stock", "Trades", "Net profit", "Win rate"], "th") + "</thead><tbody>"
                       + "".join(_row([s, str(n), f"${p:+,.2f}", f"{w:.0%}"]) for s, n, p, w in pick.iter_rows())
                       + "</tbody></table>")

    d = res["data"]
    page = TEMPLATE.format(
        title=html.escape(f"First experiment: {name}"), verdict=html.escape(verdict),
        period=f"{d['holdout_start']} → {d['holdout_end']}",
        bot_final=_usd(bot["final_equity"]), bot_ret=_pct(bot["total_return"]), bot_label=mode_names[main_mode],
        bench_final=_usd(best_bench["final_equity"]), bench_ret=_pct(best_bench["total_return"]),
        bench_label=bench_names[best_bench_key], fees=_usd(bot["fees"]), trades=f"{bot['trades']:,}",
        fees_pct=f"{bot['fees'] / cash:.0%}", win=f"{bot['win_rate']:.0%}",
        avg_gross=f"{bot['avg_trade_gross_pct'] * 100:+.3f}%", avg_net=f"{bot['avg_trade_net_pct'] * 100:+.3f}%",
        perf_table="".join(_row(r) for r in perf_rows),
        cost_head=_row(["Scenario"] + [mode_names[m] for m in modes], "th"),
        cost_table="".join(_row(r) for r in cost_rows),
        val_table="".join(_row(r) for r in val_rows), val_bench=val_bench,
        sel=f"hold {sel['horizon'] * 5} min, enter when predicted gain ≥ {sel['threshold'] * 100:.2f}%, "
            f"up to {sel['max_positions']} positions, {'extend while signal holds' if sel['extend'] else 'fixed exit'}",
        sel_ret=_pct(sel["total_return"]), sel_trades=f"{sel['trades']:,}",
        trades_table=trades_html, data_rows=f"{d['rows_5m']:,}", ds_rows=f"{d['dataset_rows']:,}",
        first=d["first_session"], last=d["last_session"], n_feat=len(res["features"]),
        cash=_usd(cash), series_json=json.dumps(series), start_cash=cash,
    )
    out = run / "report.html"
    out.write_text(page)
    return out


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fcfcfb; --surface:#ffffff; --text:#0b0b0b; --text2:#52514e; --muted:#8a8984; --grid:#e8e7e3; --border:#dddcd6;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --surface:#232322; --text:#ffffff;
  --text2:#c3c2b7; --muted:#8f8e86; --grid:#33332f; --border:#3a3a36;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --surface:#232322; --text:#ffffff; --text2:#c3c2b7; --muted:#8f8e86; --grid:#33332f;
  --border:#3a3a36; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:920px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:28px; line-height:1.2; margin:0 0 8px; }} h2 {{ font-size:20px; margin:40px 0 12px; }}
p {{ margin:0 0 12px; }} .sub {{ color:var(--text2); }} .verdict {{ font-size:20px; font-weight:600; margin:16px 0 24px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:24px; }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.tile .k {{ color:var(--text2); font-size:13px; }} .tile .v {{ font-size:26px; font-weight:650; font-variant-numeric:tabular-nums; }}
.tile .n {{ color:var(--text2); font-size:13px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; color:var(--text2); margin-bottom:8px; }}
.legend i {{ display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }}
#chart {{ position:relative; }} svg {{ display:block; width:100%; height:auto; }}
#tip {{ position:absolute; pointer-events:none; background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:8px 10px; font-size:12.5px; box-shadow:0 4px 14px rgba(0,0,0,.12); display:none; white-space:nowrap; }}
#tip b {{ display:block; margin-bottom:4px; }} #tip i {{ display:inline-block; width:10px; height:3px; margin-right:6px; vertical-align:middle; }}
#tip span {{ float:right; margin-left:14px; font-variant-numeric:tabular-nums; color:var(--text); }}
.scroll {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:14.5px; }}
th,td {{ text-align:right; padding:8px 10px; border-bottom:1px solid var(--grid); font-variant-numeric:tabular-nums; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; white-space:normal; }} th {{ color:var(--text2); font-weight:600; font-size:13px; }}
ul {{ padding-left:20px; margin:0 0 12px; }} li {{ margin-bottom:6px; }} code {{ font-size:.92em; }}
</style></head><body><main>
<h1>Can the bot beat buying an ETF?</h1>
<p class="sub">First end-to-end experiment · {cash} starting money · test period {period} · Binance Stocks fees</p>
<p class="verdict">{verdict}</p>
<div class="tiles">
  <div class="tile"><div class="k">{bot_label}</div><div class="v">{bot_final}</div><div class="n">{bot_ret} after all costs</div></div>
  <div class="tile"><div class="k">Best benchmark: {bench_label}</div><div class="v">{bench_final}</div><div class="n">{bench_ret}, one buy, then hold</div></div>
  <div class="tile"><div class="k">Fees the bot paid</div><div class="v">{fees}</div><div class="n">{fees_pct} of the starting money, over {trades} trades</div></div>
  <div class="tile"><div class="k">Average trade</div><div class="v">{avg_net}</div><div class="n">after costs ({avg_gross} before) · {win} of trades won</div></div>
</div>

<h2>Account value over the test period</h2>
<div class="card"><div class="legend" id="legend"></div><div id="chart"><svg id="svg" viewBox="0 0 880 380" role="img"
 aria-label="Account value over time for the bot and each benchmark"></svg><div id="tip"></div></div></div>

<h2>Results</h2>
<div class="card scroll"><table><thead><tr><th>Strategy</th><th>Final value</th><th>Return</th><th>Sharpe</th><th>Worst drop</th><th>Buys</th><th>Fees</th></tr></thead>
<tbody>{perf_table}</tbody></table></div>
<p class="sub" style="margin-top:8px">Sharpe is return per unit of risk (above 1 is good). Worst drop is the largest fall from a peak. Benchmarks pay the same Binance fee and spread on their one purchase.</p>

<h2>How much do costs matter?</h2>
<div class="card scroll"><table><thead>{cost_head}</thead><tbody>{cost_table}</tbody></table></div>
<p class="sub" style="margin-top:8px">Realistic costs are Binance's fee of max($0.35, 0.10%) per order, plus half the bid-ask spread and 0.01% slippage on each fill.</p>

<h2>What tuning on 2025 showed</h2>
<p>All settings were chosen on 2025 data only, using a model trained on 2021–2024. The 2026 test above was then run once.</p>
<div class="card scroll"><table><thead><tr><th>Holding time</th><th>Rank correlation</th><th>Top 1% of signals earned</th><th>Profitable settings</th><th>Best setting in 2025</th></tr></thead>
<tbody>{val_table}</tbody></table></div>
<p class="sub" style="margin-top:8px">Rank correlation measures how well predictions order the stocks (0 = random; 0.02–0.05 is typical for a usable intraday signal). "Top 1% of signals earned" is the average move after the model's strongest signals, before costs, in basis points (1 bp = 0.01%); a round trip costs about 22–30 bp.</p>
<p><b>Chosen setting:</b> {sel}. In 2025 it returned {sel_ret} over {sel_trades} trades. For comparison, buy and hold in 2025: {val_bench}.</p>

<h2>Where the bot made and lost money</h2>
<div class="card scroll">{trades_table}</div>

<h2>How the test was run</h2>
<ul>
<li><b>Data:</b> 1-minute Alpaca bars (all US exchanges) from {first} to {last}, turned into {data_rows} five-minute bars for 50 AI-chain stocks plus SPY, QQQ and SMH.</li>
<li><b>Model:</b> gradient-boosted trees on {n_feat} features (momentum, volatility, volume, time of day, market and sector moves, rank against the other stocks), predicting the price move over the holding time. {ds_rows} rows.</li>
<li><b>Trading rules:</b> decide when a 5-minute bar closes, fill at the next bar's open, buy only, at least $350 per order, no new positions in the last 30 minutes, everything sold before the close.</li>
<li><b>No peeking:</b> trained on 2021–2024, tuned on 2025, tested once on 2026. "Retrained monthly" retrains at each month start on all earlier data.</li>
</ul>
<h2>Limits of this result</h2>
<ul>
<li>The 50 stocks were picked in 2026 knowing they were AI winners, which flatters the basket benchmark and any strategy trading them.</li>
<li>Spreads are assumed by size tier, not measured. Binance's real fills may differ.</li>
<li>The base case assumes money from a sale can be reused the same day. Binance's settlement rule is unconfirmed; the "usable next day" row shows the other case.</li>
<li>One model family and one test period. A single result, good or bad, is weak evidence.</li>
</ul>
</main>
<script>
const S={series_json}, START={start_cash};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const svg=document.getElementById('svg'), tip=document.getElementById('tip'), NS='http://www.w3.org/2000/svg';
const W=880,H=380,m={{l:56,r:16,t:12,b:30}}, dates=S[0].dates, n=dates.length;
const all=S.flatMap(s=>s.values).concat([START]); let lo=Math.min(...all), hi=Math.max(...all);
const pad=(hi-lo)*0.06; lo-=pad; hi+=pad;
const x=i=>m.l+(W-m.l-m.r)*i/(n-1), y=v=>m.t+(H-m.t-m.b)*(1-(v-lo)/(hi-lo));
const el=(t,a,p=svg)=>{{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);p.appendChild(e);return e;}};
function draw(){{
  svg.innerHTML=''; const step=Math.pow(10,Math.floor(Math.log10(hi-lo)))/2;
  for(let v=Math.ceil(lo/step)*step; v<=hi; v+=step){{
    el('line',{{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:css('--grid'),'stroke-width':1}});
    el('text',{{x:m.l-8,y:y(v)+4,'text-anchor':'end','font-size':12,fill:css('--muted')}}).textContent='$'+v.toLocaleString();
  }}
  el('line',{{x1:m.l,x2:W-m.r,y1:y(START),y2:y(START),stroke:css('--muted'),'stroke-width':1}});
  let last='';
  dates.forEach((d,i)=>{{const mo=d.slice(0,7); if(mo!==last){{last=mo; if(i>0||true) el('text',{{x:x(i),y:H-8,'font-size':12,
    fill:css('--muted'),'text-anchor':i===0?'start':'middle'}}).textContent=new Date(d+'T00:00').toLocaleString('en',{{month:'short'}});}}}});
  S.forEach(s=>el('path',{{d:s.values.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(''),fill:'none',
    stroke:css('--s'+s.slot),'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'}}));
  cross=el('line',{{y1:m.t,y2:H-m.b,stroke:css('--muted'),'stroke-width':1,visibility:'hidden'}});
}}
let cross; draw();
document.getElementById('legend').innerHTML=S.map(s=>`<span><i style="background:var(--s${{s.slot}})"></i>${{s.label}}</span>`).join('');
svg.addEventListener('pointermove',e=>{{
  const r=svg.getBoundingClientRect(), px=(e.clientX-r.left)*W/r.width;
  const i=Math.max(0,Math.min(n-1,Math.round((px-m.l)/(W-m.l-m.r)*(n-1))));
  cross.setAttribute('x1',x(i)); cross.setAttribute('x2',x(i)); cross.setAttribute('visibility','visible');
  const rows=[...S].sort((a,b)=>b.values[i]-a.values[i]).map(s=>`<div><i style="background:var(--s${{s.slot}})"></i>${{s.label}}<span>$${{s.values[i].toLocaleString(undefined,{{maximumFractionDigits:0}})}}</span></div>`).join('');
  tip.innerHTML=`<b>${{dates[i]}}</b>${{rows}}`; tip.style.display='block';
  const left=x(i)*r.width/W; tip.style.left=(left>r.width/2?left-tip.offsetWidth-12:left+12)+'px'; tip.style.top='8px';
}});
svg.addEventListener('pointerleave',()=>{{tip.style.display='none'; cross.setAttribute('visibility','hidden');}});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
</script></body></html>
"""
