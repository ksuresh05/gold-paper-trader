"""
Gold (XAUUSD) 9/20/EMA100 Paper Trading Tracker
---------------------------------------------------------------------
PURPOSE: Runs the SAME validated signal logic from gold_v1_long_history_1h.py
(9/20 EMA cross + pullback entry, trend filter vs EMA20, fixed stop at
EMA100 at entry, exit on stop hit or opposite 9/20 cross) against live
1-hour gold data, on a schedule (run via Windows Task Scheduler, same
pattern as your Upstox auto-start).

This is a TRACKER, not an order-placer: it tells you what to do (enter
LONG/SHORT, or exit an open position) and you place/close the trade
yourself on your Pepperstone cTrader demo account. It maintains its own
persistent state file (state/paper_state.json) so each hourly run picks
up where the last one left off -- open position, account balance, full
trade history.

RULES (identical to the validated backtest):
  Entry: 9 EMA crosses 20 EMA -> setup ON -> price pulls back and touches
         the 9 EMA -> trend filter (Close vs EMA20) -> enter at Close
  Stop:  fixed at the EMA100 value captured at entry time, never trails
  Exit:  stop hit, OR 9 EMA crosses back the other way vs 20 EMA
         (whichever the most recent completed 1h bar shows)
  Sizing: 2% risk per trade, based on account balance CAPPED AT $5,000
          for sizing purposes (position size stops growing past that
          equity level, but trading continues past $5,000 real balance)
  Leverage: 1:200, capped at 20% of theoretical margin as a safety buffer
  No withdrawal logic -- pure compounding (up to the $5,000 sizing cap)

SETUP (Windows Task Scheduler, matching your UpstoxTradingBot pattern):
  1. This script should run once per hour, a few minutes after each new
     1h gold candle closes (e.g. at :05 past the hour) so the latest
     completed bar is available from the data feed.
  2. Use run_paper_trader.bat to launch it (see that file for the exact
     command). Point Task Scheduler at the .bat file, trigger hourly.
  3. Each run appends to state/paper_state.json and state/trade_log.csv.
     Check html/paper_trading_report.html after each run, or open it
     any time to see current status and full trade history.

Run manually to test:
    pip install yfinance pandas numpy
    python3 gold_paper_trader.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime, timezone

# ----------------------------- CONFIG ---------------------------------
TICKER = "GC=F"
FALLBACK_TICKERS = ["MGC=F", "GLD"]
INTERVAL = "1h"
PERIOD = "60d"   # only need recent bars for live signal generation, not the full history
EMA_FAST = 7
EMA_SLOW = 15
EMA_STOP = 150
ATR_PERIOD = 14   # kept for reference/diagnostics even though USE_ATR_TRAIL=False

# ------------------------- SESSION FILTER (UNTESTED vs the backtest) ------------------------
# The validated backtest ran on ALL hours, no session restriction. This filter is a new,
# not-yet-validated rule being tested during paper trading: only take NEW entries during
# London+NY overlap hours, on the theory that thinner Asian-session liquidity produces more
# false crosses / noise-driven stop-outs (consistent with the tight-stop fragility flagged
# repeatedly by the backtest's sanity checks). Set SESSION_FILTER_ENABLED=False to revert to
# the exact validated 24/5 behavior for comparison.
SESSION_FILTER_ENABLED = False
SESSION_START_HOUR_GMT = 8    # London open, approx
SESSION_END_HOUR_GMT = 21     # NY afternoon, approx -- covers London+NY overlap and NY session

STATE_DIR = "state"
STATE_FILE = os.path.join(STATE_DIR, "paper_state.json")
TRADE_LOG_CSV = os.path.join(STATE_DIR, "trade_log.csv")
HTML_DIR = "html"
OUTPUT_HTML = os.path.join(HTML_DIR, "paper_trading_report.html")

# ------------------------- ACCOUNT / RISK CONFIG ------------------------
STARTING_CAPITAL = 1000.0
RISK_PCT = 2.0
SIZING_EQUITY_CAP = 5000.0   # position sizing uses min(actual_balance, this cap) --
                              # so growth compounds normally below $5,000, and position
                              # size stops increasing further above it, but trading
                              # and compounding continue past $5,000 real balance.

# ------------------------- PROFIT WITHDRAWAL CONFIG (matches the backtest's cap mode) ------------------------
WITHDRAWAL_ENABLED = True
WITHDRAWAL_CAP_LEVEL = 5000.0   # once balance would exceed this on a winning trade, everything
                                  # above it is withdrawn immediately -- the trading account itself
                                  # never compounds past this level again, though trading continues
                                  # at that fixed size. Matches gold_v1_long_history_1h.py's
                                  # WITHDRAWAL_MODE="cap" behavior exactly, so paper results are
                                  # comparable to the validated 7/15/150 backtest.

# ------------------------- PEPPERSTONE MARGIN / LOT CONFIG ------------------------
PEPPERSTONE_LOT_SIZE_OZ = 100.0
PEPPERSTONE_MIN_LOT = 0.01
PEPPERSTONE_LEVERAGE = 200
MARGIN_SAFETY_FRACTION = 0.20
ROUND_TRIP_COST_PER_OZ = 1.00
MIN_STOP_DIST_USD = 0.50


def _try_fetch(ticker, retries=2, backoff_sec=3):
    import time
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"  {ticker} attempt {attempt+1} raised: {e}")
        if attempt < retries - 1:
            time.sleep(backoff_sec)
    return pd.DataFrame()


def fetch_data():
    tickers_to_try = [TICKER] + FALLBACK_TICKERS
    for ticker in tickers_to_try:
        print(f"Trying {ticker} ({INTERVAL} interval, {PERIOD} period)...")
        df = _try_fetch(ticker)
        if not df.empty:
            if ticker != TICKER:
                print(f"NOTE: {TICKER} was unavailable -- using fallback ticker {ticker} instead.")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=str.title)
            df = df.dropna()
            return df
    raise RuntimeError(
        f"No data returned for {TICKER} or any fallback ({FALLBACK_TICKERS}). "
        f"Try again shortly, or check your network connection."
    )


def add_indicators(df):
    df["EMA_FAST"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_SLOW"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["EMA_STOP"] = df["Close"].ewm(span=EMA_STOP, adjust=False).mean()

    diff = df["EMA_FAST"] - df["EMA_SLOW"]
    prev_diff = diff.shift()
    df["CROSS"] = 0
    df.loc[(prev_diff <= 0) & (diff > 0), "CROSS"] = 1
    df.loc[(prev_diff >= 0) & (diff < 0), "CROSS"] = -1

    return df.dropna(subset=["EMA_FAST", "EMA_SLOW", "EMA_STOP"])


def _in_session_window(ts):
    """
    Checks whether a bar's timestamp falls within the allowed session window
    (SESSION_START_HOUR_GMT to SESSION_END_HOUR_GMT), converting to GMT/UTC
    first regardless of the timestamp's original timezone. Only gates NEW
    entries -- an already-open position is managed/exited regardless of
    session, since abandoning risk management because the clock struck 9pm
    would be worse than the session filter itself.
    """
    ts_gmt = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    hour = ts_gmt.hour
    return SESSION_START_HOUR_GMT <= hour < SESSION_END_HOUR_GMT


# ------------------------- STATE MANAGEMENT ------------------------

def load_state():
    """
    State schema:
    {
      "account_balance": float,
      "open_position": null | {
          "direction": "LONG"/"SHORT", "entry_time": iso str, "entry_price": float,
          "stop_price": float, "stop_dist_usd": float, "position_oz": float,
          "position_lots": float
      },
      "setup_direction": 0/1/-1,   # tracks "cross happened, waiting for pullback touch"
                                    # across runs, same as the backtest's setup_direction
      "last_processed_bar": iso str or null,  # the last 1h bar timestamp already acted on,
                                                 # prevents reprocessing the same bar twice
      "trade_count": int
    }
    """
    if not os.path.exists(STATE_FILE):
        return {
            "account_balance": STARTING_CAPITAL,
            "open_position": None,
            "setup_direction": 0,
            "last_processed_bar": None,  # None is a sentinel meaning "never initialized" --
                                          # main()/run_once() fills this in on the very first
                                          # run using the latest available bar at that moment,
                                          # so a fresh start never replays historical signals.
            "trade_count": 0,
            "total_withdrawn": 0.0,
            "initialized": False  # flips to True once the fresh-start bootstrap has run once
        }
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    state.setdefault("total_withdrawn", 0.0)  # migration for state files saved before this field existed
    return state


def save_state(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_trade_log(row_dict):
    """
    Upserts by entry_time (unique per trade, since only one position is
    open at a time). An ENTRY writes a new OPEN row; the matching EXIT
    later updates that SAME row to CLOSED with exit details, rather than
    appending a second row for the same trade.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    log_df = read_trade_log()

    if not log_df.empty and (log_df["entry_time"].astype(str) == str(row_dict["entry_time"])).any():
        idx = log_df.index[log_df["entry_time"].astype(str) == str(row_dict["entry_time"])][0]
        # Build the updated row as a fresh Series with object dtype and
        # reassign the whole row at once -- avoids per-cell dtype coercion
        # errors when a column that was previously empty/blank (inferred
        # as float64 by pandas) needs to hold a string value like a
        # timestamp on close.
        for col in log_df.columns:
            if col not in row_dict:
                row_dict[col] = log_df.loc[idx, col]
        log_df = log_df.astype(object)
        log_df.loc[idx] = pd.Series(row_dict)
    else:
        log_df = pd.concat([log_df.astype(object) if not log_df.empty else log_df,
                             pd.DataFrame([row_dict])], ignore_index=True)

    log_df.to_csv(TRADE_LOG_CSV, mode="w", header=True, index=False)


def read_trade_log():
    if os.path.exists(TRADE_LOG_CSV):
        return pd.read_csv(TRADE_LOG_CSV)
    return pd.DataFrame(columns=[
        "entry_time", "exit_time", "direction", "status", "entry_price", "exit_price",
        "stop_price", "position_oz", "position_lots", "gross_pnl_usd", "cost_usd",
        "net_pnl_usd", "account_balance", "withdrawal_usd", "total_withdrawn"
    ])


# ------------------------- SIZING ------------------------

def max_position_oz(gold_price, sizing_equity):
    margin_per_oz = gold_price / PEPPERSTONE_LEVERAGE
    theoretical_max_oz = sizing_equity / margin_per_oz
    return theoretical_max_oz * MARGIN_SAFETY_FRACTION


def round_to_lot_step(oz):
    lots = oz / PEPPERSTONE_LOT_SIZE_OZ
    lots_rounded = (lots // PEPPERSTONE_MIN_LOT) * PEPPERSTONE_MIN_LOT
    return lots_rounded * PEPPERSTONE_LOT_SIZE_OZ


def compute_position_size(account_balance, stop_dist_usd, gold_price):
    sizing_equity = min(account_balance, SIZING_EQUITY_CAP)
    risk_dollars = sizing_equity * (RISK_PCT / 100)
    position_oz = risk_dollars / stop_dist_usd
    margin_cap_oz = max_position_oz(gold_price, sizing_equity)
    position_oz = min(position_oz, margin_cap_oz)
    position_oz = round_to_lot_step(position_oz)
    return position_oz


# ------------------------- SIGNAL PROCESSING ------------------------

def process_new_bars(df, state):
    """
    Walks forward through any 1h bars not yet processed (per state's
    last_processed_bar), applying the exact same entry/exit logic as the
    validated backtest, one completed bar at a time. Only ever acts on
    fully CLOSED bars -- the most recent row from yfinance may be a
    still-forming current-hour bar, which is excluded to avoid acting on
    incomplete data (mirrors how a live trader would only react after a
    candle closes, not mid-candle).

    Returns a list of "events" (dicts) describing what happened on each
    processed bar -- new entry, exit, or nothing -- for the caller to log
    and report. Mutates state in place.
    """
    events = []

    now_utc = pd.Timestamp.now(tz="UTC")
    # Exclude the current still-forming bar: keep only bars whose hour has
    # fully elapsed. df.index is tz-aware (from yfinance); compare in UTC.
    df_closed = df[df.index.tz_convert("UTC") + pd.Timedelta(hours=1) <= now_utc]

    if state["last_processed_bar"] is not None:
        last_ts = pd.Timestamp(state["last_processed_bar"])
        df_closed = df_closed[df_closed.index > last_ts]

    if df_closed.empty:
        return events  # nothing new to process since last run

    for ts, row in df_closed.iterrows():
        pos = state["open_position"]

        if pos is not None:
            d = 1 if pos["direction"] == "LONG" else -1
            stop_price = pos["stop_price"]
            hit_stop = (d == 1 and row["Low"] <= stop_price) or \
                       (d == -1 and row["High"] >= stop_price)
            opposite_cross = row["CROSS"] == (-d)

            if hit_stop or opposite_cross:
                exit_price = stop_price if hit_stop else row["Close"]
                result = "WIN" if (exit_price - pos["entry_price"]) * d > 0 else "LOSS"

                position_oz = pos["position_oz"]
                gross_pnl = (exit_price - pos["entry_price"]) * position_oz
                if pos["direction"] == "SHORT":
                    gross_pnl = (pos["entry_price"] - exit_price) * position_oz
                cost = ROUND_TRIP_COST_PER_OZ * position_oz
                net_pnl = gross_pnl - cost

                state["account_balance"] += net_pnl
                state["trade_count"] += 1

                # Profit cap: once balance exceeds WITHDRAWAL_CAP_LEVEL, withdraw
                # everything above it -- matches the validated backtest's
                # WITHDRAWAL_MODE="cap" exactly, so paper results stay
                # comparable to that backtest.
                withdrawal_this_trade = 0.0
                if WITHDRAWAL_ENABLED and state["account_balance"] > WITHDRAWAL_CAP_LEVEL:
                    withdrawal_this_trade = state["account_balance"] - WITHDRAWAL_CAP_LEVEL
                    state["account_balance"] = WITHDRAWAL_CAP_LEVEL
                    state["total_withdrawn"] += withdrawal_this_trade

                trade_record = {
                    "entry_time": pos["entry_time"], "exit_time": str(ts),
                    "direction": pos["direction"], "status": "CLOSED",
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "stop_price": round(stop_price, 2),
                    "position_oz": round(position_oz, 2),
                    "position_lots": round(pos["position_lots"], 2),
                    "gross_pnl_usd": round(gross_pnl, 2),
                    "cost_usd": round(cost, 2),
                    "net_pnl_usd": round(net_pnl, 2),
                    "account_balance": round(state["account_balance"], 2),
                    "withdrawal_usd": round(withdrawal_this_trade, 2),
                    "total_withdrawn": round(state["total_withdrawn"], 2)
                }
                append_trade_log(trade_record)
                events.append({"type": "EXIT", "reason": "STOP" if hit_stop else "OPPOSITE_CROSS",
                                "result": result, **trade_record})

                state["open_position"] = None
                state["setup_direction"] = 0
            # if position still open after this bar, fall through to next bar
            # without evaluating a new entry (matches backtest's `continue`)
            continue

        # No open position -- look for a new setup/entry, same logic as backtest
        if row["CROSS"] == 1:
            state["setup_direction"] = 1
        elif row["CROSS"] == -1:
            state["setup_direction"] = -1

        if state["setup_direction"] != 0:
            touched = (row["Low"] <= row["EMA_FAST"] <= row["High"])
            if touched:
                trend_ok = (state["setup_direction"] == 1 and row["Close"] > row["EMA_SLOW"]) or \
                           (state["setup_direction"] == -1 and row["Close"] < row["EMA_SLOW"])
                if not trend_ok:
                    state["setup_direction"] = 0
                elif SESSION_FILTER_ENABLED and not _in_session_window(ts):
                    # Setup and trend filter both passed, but outside the allowed session
                    # window -- skip this entry (untested filter, see config comment above).
                    events.append({"type": "SIGNAL_SKIPPED", "reason": "outside_session_window",
                                    "entry_time": str(ts)})
                    state["setup_direction"] = 0
                else:
                    entry_price = row["Close"]
                    trade_dir = state["setup_direction"]
                    initial_stop = row["EMA_STOP"]
                    stop_dist = abs(entry_price - initial_stop)

                    if stop_dist < MIN_STOP_DIST_USD:
                        # Same data-quality filter as the backtest -- too tight to size sanely
                        events.append({"type": "SIGNAL_SKIPPED", "reason": "stop_too_tight",
                                        "entry_time": str(ts), "stop_dist_usd": round(stop_dist, 4)})
                        state["setup_direction"] = 0
                    else:
                        position_oz = compute_position_size(state["account_balance"], stop_dist, entry_price)
                        if position_oz <= 0:
                            events.append({"type": "SIGNAL_SKIPPED", "reason": "sizes_to_zero_lots",
                                            "entry_time": str(ts)})
                            state["setup_direction"] = 0
                        else:
                            state["open_position"] = {
                                "direction": "LONG" if trade_dir == 1 else "SHORT",
                                "entry_time": str(ts), "entry_price": entry_price,
                                "stop_price": initial_stop, "stop_dist_usd": stop_dist,
                                "position_oz": position_oz,
                                "position_lots": position_oz / PEPPERSTONE_LOT_SIZE_OZ
                            }
                            state["setup_direction"] = 0

                            trade_record = {
                                "entry_time": str(ts), "exit_time": "",
                                "direction": "LONG" if trade_dir == 1 else "SHORT",
                                "status": "OPEN",
                                "entry_price": round(entry_price, 2), "exit_price": "",
                                "stop_price": round(initial_stop, 2),
                                "position_oz": round(position_oz, 2),
                                "position_lots": round(position_oz / PEPPERSTONE_LOT_SIZE_OZ, 2),
                                "gross_pnl_usd": "", "cost_usd": "", "net_pnl_usd": "",
                                "account_balance": round(state["account_balance"], 2),
                                "withdrawal_usd": 0.0, "total_withdrawn": round(state["total_withdrawn"], 2)
                            }
                            append_trade_log(trade_record)
                            events.append({"type": "ENTRY", **trade_record})

        state["last_processed_bar"] = str(ts)

    return events


# ------------------------- REPORTING ------------------------

def build_html_report(state, events, trade_log_df):
    total_withdrawn = state.get("total_withdrawn", 0.0)
    total_wealth = state["account_balance"] + total_withdrawn
    net_return_pct = (state["account_balance"] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    total_wealth_return_pct = (total_wealth - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    open_pos = state["open_position"]
    open_pos_block = ""
    if open_pos:
        d_color = "#16a34a" if open_pos["direction"] == "LONG" else "#dc2626"
        open_pos_block = f"""
    <div class="section-title">Open Position</div>
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-label">Direction</div><div class="stat-value" style="color:{d_color}">{open_pos['direction']}</div></div>
        <div class="stat-card"><div class="stat-label">Entry Time</div><div class="stat-value" style="font-size:14px">{open_pos['entry_time']}</div></div>
        <div class="stat-card"><div class="stat-label">Entry Price</div><div class="stat-value">${open_pos['entry_price']:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Stop Price</div><div class="stat-value">${open_pos['stop_price']:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Size</div><div class="stat-value">{open_pos['position_oz']:.2f} oz ({open_pos['position_lots']:.2f} lot)</div></div>
    </div>"""
    else:
        open_pos_block = """
    <div class="section-title">Open Position</div>
    <div class="verdict" style="border-left-color:#6b7280;">No open position -- scanning for the next setup.</div>"""

    events_block = ""
    if events:
        event_items = ""
        for e in events:
            if e["type"] == "ENTRY":
                event_items += f"<li>ENTRY: {e['direction']} at ${e['entry_price']:.2f}, bar {e['entry_time']}</li>"
            elif e["type"] == "EXIT":
                color = "#16a34a" if e["result"] == "WIN" else "#dc2626"
                event_items += (f"<li style='color:{color}'>EXIT ({e['reason']}, {e['result']}): "
                                 f"{e['direction']} closed at ${e['exit_price']:.2f}, net ${e['net_pnl_usd']:.2f}, "
                                 f"bar {e['exit_time']}</li>")
            elif e["type"] == "SIGNAL_SKIPPED":
                event_items += f"<li style='color:#9ca3af'>Signal skipped ({e['reason']}) at bar {e['entry_time']}</li>"
        events_block = f"""
    <div class="section-title">Events This Run</div>
    <ul style="font-size:13px; line-height:1.8;">{event_items}</ul>"""
    else:
        events_block = """
    <div class="section-title">Events This Run</div>
    <div class="verdict" style="border-left-color:#6b7280;">No new completed bars to process since the last run.</div>"""

    trade_rows = ""
    if not trade_log_df.empty:
        for _, t in trade_log_df.sort_values("entry_time", ascending=False).iterrows():
            status_color = "#f59e0b" if t["status"] == "OPEN" else \
                            ("#16a34a" if str(t.get("net_pnl_usd", "")) not in ("", "nan") and float(t["net_pnl_usd"]) > 0 else "#dc2626")
            dir_color = "#16a34a" if t["direction"] == "LONG" else "#dc2626"
            exit_price_str = f"${float(t['exit_price']):.2f}" if str(t["exit_price"]) not in ("", "nan") else "-"
            net_pnl_str = f"${float(t['net_pnl_usd']):.2f}" if str(t["net_pnl_usd"]) not in ("", "nan") else "-"
            withdrawal_val = t.get("withdrawal_usd", 0.0)
            withdrawal_str = (f"${float(withdrawal_val):.2f}"
                               if str(withdrawal_val) not in ("", "nan") and float(withdrawal_val) > 0 else "-")
            trade_rows += f"""
        <tr>
            <td>{t['entry_time']}</td>
            <td style="color:{dir_color};font-weight:600">{t['direction']}</td>
            <td style="color:{status_color};font-weight:600">{t['status']}</td>
            <td>${float(t['entry_price']):.2f}</td>
            <td>{exit_price_str}</td>
            <td>{t['position_oz']:.2f} oz ({t['position_lots']:.2f} lot)</td>
            <td style="color:{status_color};font-weight:600">{net_pnl_str}</td>
            <td>${float(t['account_balance']):.2f}</td>
            <td style="color:#f59e0b">{withdrawal_str}</td>
        </tr>"""

    closed = trade_log_df[trade_log_df["status"] == "CLOSED"] if not trade_log_df.empty else pd.DataFrame()
    wins = (closed["net_pnl_usd"].astype(float) > 0).sum() if not closed.empty else 0
    win_rate = round(wins / len(closed) * 100, 1) if len(closed) else 0

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Gold Paper Trading Tracker</title>
<style>
    body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background:#0f1115; color:#e5e7eb; margin:0; padding:24px; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ font-size: 22px; margin-bottom:4px; }}
    .subtitle {{ color:#9ca3af; font-size:13px; margin-bottom:24px; }}
    .warning {{ background:#3f2d0f; border:1px solid #92620a; color:#fbbf24; padding:12px 16px; border-radius:8px; font-size:13px; margin-bottom:24px; }}
    .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:28px; }}
    .stat-card {{ background:#1a1d24; border:1px solid #2a2d36; border-radius:10px; padding:14px; }}
    .stat-label {{ font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:0.5px; }}
    .stat-value {{ font-size:22px; font-weight:700; margin-top:4px; }}
    .verdict {{ background:#1a1d24; border-left:4px solid #16a34a; padding:14px 16px; border-radius:6px; margin-bottom:28px; font-size:14px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; margin-bottom: 20px;}}
    th {{ text-align:left; padding:10px; background:#1a1d24; color:#9ca3af; font-weight:600; border-bottom:1px solid #2a2d36; }}
    td {{ padding:8px 10px; border-bottom:1px solid #22252c; }}
    .section-title {{ font-size:15px; font-weight:600; margin:28px 0 12px; }}
</style></head>
<body>
<div class="container">
    <h1>Gold (XAUUSD) Paper Trading Tracker</h1>
    <div class="subtitle">EMA{EMA_FAST}/{EMA_SLOW} cross + pullback, EMA{EMA_STOP} fixed stop, {RISK_PCT:.0f}% risk, 1:{PEPPERSTONE_LEVERAGE} leverage @ {MARGIN_SAFETY_FRACTION*100:.0f}% safety cap</div>
    <div class="subtitle">{'Session filter ON: new entries only ' + str(SESSION_START_HOUR_GMT) + '-' + str(SESSION_END_HOUR_GMT) + ' GMT (untested vs backtest -- London/NY hours)' if SESSION_FILTER_ENABLED else 'Session filter OFF: 24/5, matches validated backtest exactly'}</div>
    <div class="subtitle" style="color:#f59e0b; font-weight:600;">Config: EMA_FAST={EMA_FAST}, EMA_SLOW={EMA_SLOW}, EMA_STOP={EMA_STOP}, STARTING_CAPITAL=${STARTING_CAPITAL:.0f}, SIZING_EQUITY_CAP=${SIZING_EQUITY_CAP:,.0f} -- check this matches the backtest you're comparing against</div>
    <div class="subtitle">Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>

    <div class="warning">
        ⚠ This is a PAPER TRACKER, not an auto-trader. It tells you what to do based on completed
        1h candles -- place/close the actual trade yourself on your Pepperstone cTrader demo account.
        Position sizing is capped for calculation purposes at ${SIZING_EQUITY_CAP:,.0f} equity -- growth
        continues past that, but position size stops increasing further once real balance exceeds it.
    </div>

    <div class="section-title">Account Status</div>
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-label">Account Balance</div><div class="stat-value">${state['account_balance']:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Total Withdrawn</div><div class="stat-value">${total_withdrawn:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Total Wealth Return</div><div class="stat-value">{total_wealth_return_pct:.2f}%</div></div>
        <div class="stat-card"><div class="stat-label">Total Trades</div><div class="stat-value">{state['trade_count']}</div></div>
        <div class="stat-card"><div class="stat-label">Win Rate (closed trades)</div><div class="stat-value">{win_rate}%</div></div>
    </div>

    {open_pos_block}

    {events_block}

    <div class="section-title">Full Trade History</div>
    <table>
        <tr><th>Entry Time</th><th>Direction</th><th>Status</th><th>Entry</th><th>Exit</th><th>Size</th><th>Net P&L</th><th>Account Balance</th><th>Withdrawal</th></tr>
        {trade_rows if trade_rows else '<tr><td colspan="8">No trades yet.</td></tr>'}
    </table>
</div>
</body></html>"""
    return html


def run_once():
    """
    Performs a single check-and-act cycle: fetch data, process any newly
    closed bars, save state, write the report. Called repeatedly by the
    main loop below.
    """
    print(f"\n=== Gold Paper Trader check at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
    print(f"Config: EMA_FAST={EMA_FAST}, EMA_SLOW={EMA_SLOW}, EMA_STOP={EMA_STOP}, "
          f"STARTING_CAPITAL=${STARTING_CAPITAL:.0f}, RISK_PCT={RISK_PCT:.0f}%, "
          f"SIZING_EQUITY_CAP=${SIZING_EQUITY_CAP:,.0f}")
    if SESSION_FILTER_ENABLED:
        print(f"Session filter: ON -- new entries only {SESSION_START_HOUR_GMT}:00-{SESSION_END_HOUR_GMT}:00 GMT "
              f"(untested vs the validated backtest, which ran 24/5)")
    else:
        print("Session filter: OFF -- 24/5, matches validated backtest exactly")
    state = load_state()
    print(f"Loaded state: balance=${state['account_balance']:.2f}, "
          f"open_position={'YES - ' + state['open_position']['direction'] if state['open_position'] else 'None'}, "
          f"trades so far={state['trade_count']}")

    df = fetch_data()
    print(f"Fetched {len(df)} bars, latest: {df.index.max()}")
    df = add_indicators(df)

    if not state.get("initialized", False):
        # Fresh start (true "Day 1"): mark every bar up to and including the
        # latest one CURRENTLY available as already-seen, so process_new_bars
        # only ever acts on bars that close AFTER this exact moment. This is
        # what makes the tracker forward-only from the instant it first runs,
        # instead of replaying weeks of historical signals as if they just
        # happened (the original bootstrap bug).
        state["last_processed_bar"] = str(df.index.max())
        state["initialized"] = True
        print(f"FRESH START: marking all bars up to {df.index.max()} as already-seen. "
              f"Only NEW bars closing after this point will generate trades from here on.")

    events = process_new_bars(df, state)
    save_state(state)

    if events:
        print(f"\n{len(events)} event(s) this check:")
        for e in events:
            if e["type"] == "ENTRY":
                print(f"  -> ENTER {e['direction']} at ${e['entry_price']:.2f}, "
                      f"size {e['position_oz']:.2f} oz ({e['position_lots']:.2f} lot), "
                      f"stop ${e['stop_price']:.2f}, bar {e['entry_time']}")
            elif e["type"] == "EXIT":
                print(f"  -> EXIT ({e['reason']}, {e['result']}): {e['direction']} closed at "
                      f"${e['exit_price']:.2f}, net ${e['net_pnl_usd']:.2f}, balance now "
                      f"${e['account_balance']:.2f}, bar {e['exit_time']}")
            elif e["type"] == "SIGNAL_SKIPPED":
                print(f"  -> Signal skipped ({e['reason']}) at bar {e['entry_time']}")
    else:
        print("No new completed bars since last check -- nothing to process.")

    print(f"Current balance: ${state['account_balance']:.2f} | "
          f"Open position: {state['open_position']['direction'] if state['open_position'] else 'None'}")

    trade_log_df = read_trade_log()
    html = build_html_report(state, events, trade_log_df)
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report updated: {OUTPUT_HTML}")


def _seconds_until_next_window_open():
    """
    When outside the active GMT window, computes how long to sleep before
    the window next opens (today if it hasn't started yet, tomorrow if
    today's window has already closed).
    """
    now = datetime.now(timezone.utc)
    today_open = now.replace(hour=SESSION_START_HOUR_GMT, minute=0, second=0, microsecond=0)
    if now < today_open:
        target = today_open
    else:
        # today's window already ended (or hasn't started but this branch
        # only hits if now >= today_open, so it must have ended) -- next
        # opening is tomorrow
        target = today_open + pd.Timedelta(days=1)
    return max((target - now).total_seconds(), 0)


def main_continuous():
    """
    Continuous loop, started once (e.g. via a scheduled task at machine
    startup or once each morning) and left running -- same pattern as the
    Upstox ratchet bot's intraday loop. Active only during the configured
    GMT session window; outside that window it sleeps until the window
    next opens rather than exiting, so it can be left running unattended.
    An open position is still monitored for exits even though new entries
    are session-gated -- risk management never pauses.

    POLL_INTERVAL_SECONDS controls how often it checks for a newly closed
    1h candle while active.

    Requires a host that stays continuously awake (a paid always-on task,
    or a machine that never sleeps). NOT suitable for a free-tier daily
    scheduled task or a laptop that sleeps -- use main_once() for those.
    """
    POLL_INTERVAL_SECONDS = 20 * 60  # 20 minutes, within the requested 15-30 min range

    print("=== Gold Paper Trader -- continuous mode ===")
    print(f"Active window: {SESSION_START_HOUR_GMT}:00-{SESSION_END_HOUR_GMT}:00 GMT, "
          f"polling every {POLL_INTERVAL_SECONDS // 60} minutes while active.")
    print("Leave this window running. Press Ctrl+C to stop.\n")

    import time
    while True:
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        if SESSION_START_HOUR_GMT <= current_hour < SESSION_END_HOUR_GMT:
            try:
                run_once()
            except Exception as e:
                # Never let one failed check (e.g. a transient data-fetch
                # error) kill the whole session -- log it and keep going,
                # since an open position still needs to be monitored on
                # the next cycle.
                print(f"ERROR during check: {e}")
                print("Will retry on the next cycle.")
            print(f"Sleeping {POLL_INTERVAL_SECONDS // 60} minutes until next check...")
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            sleep_secs = _seconds_until_next_window_open()
            wake_time = now + pd.Timedelta(seconds=sleep_secs)
            print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Outside active window "
                  f"({SESSION_START_HOUR_GMT}:00-{SESSION_END_HOUR_GMT}:00 GMT). "
                  f"Sleeping until {wake_time.strftime('%Y-%m-%d %H:%M UTC')} "
                  f"({sleep_secs/3600:.1f} hours)...")
            # Sleep in chunks so a long idle period can still be interrupted
            # cleanly (Ctrl+C) rather than one giant uninterruptible sleep.
            remaining = sleep_secs
            chunk = 300  # check in every 5 minutes even while idle
            while remaining > 0:
                time.sleep(min(chunk, remaining))
                remaining -= chunk


def main_once():
    """
    Single-check mode: runs exactly one run_once() call and exits
    immediately. Designed for PythonAnywhere's FREE tier, which only
    supports scheduled tasks (once daily), not always-on continuous
    processes. Also fits any environment that can't stay awake 24/7
    (e.g. a laptop you don't want running continuously).

    IMPORTANT TRADEOFF vs continuous mode: process_new_bars() walks
    forward through ALL bars closed since the last check, applying the
    exact same entry/exit logic to each one in sequence -- so no signals
    are silently skipped by checking only once a day. However, an
    open position's stop can only be detected as hit up to once per day
    (whenever this next runs), not within the same hour it actually
    happened -- a real, accepted tradeoff for zero hosting cost. If a
    day's price action would have hit the stop AND later would have
    triggered the opposite-cross exit before the next daily check, only
    the stop-hit (checked first, bar by bar, exactly as it occurred) is
    recorded -- the bar-by-bar walk in process_new_bars() still resolves
    events in the correct chronological order, this just means you find
    out about them up to a day late rather than within 20 minutes.
    """
    print("=== Gold Paper Trader -- single-check mode (for daily scheduled tasks) ===")
    try:
        run_once()
    except Exception as e:
        print(f"ERROR during check: {e}")
        raise  # in single-run mode, let the scheduler's own error/retry handling see this
    print("Single check complete. Exiting (will run again at the next scheduled time).")


def main_bounded_loop(loop_minutes, check_interval_minutes, git_commit_each_check):
    """
    Runs a time-BOUNDED loop (not infinite): checks every
    check_interval_minutes, for up to loop_minutes total, then exits
    cleanly. Designed for GitHub Actions: a single workflow run stays
    alive for a fixed duration (intentionally longer than the trigger
    interval, e.g. 70 min on an hourly trigger) so that even if the NEXT
    scheduled trigger is delayed, this run is still actively checking
    and covering the gap.

    If git_commit_each_check is True, commits+pushes state/html after
    EVERY individual check (not just once at the end) -- so if the job
    is killed early (crash, GitHub's own timeout), only the single
    in-progress check is lost, not the whole run's accumulated work.
    Requires git to be configured (user.name/user.email) and the
    workflow to have write permission -- both handled by the calling
    workflow, not this function.

    Unlike main_continuous(), this does NOT gate on SESSION_START_HOUR_GMT/
    SESSION_END_HOUR_GMT sleep-until-window logic -- it runs for its
    fixed duration regardless of hour, since SESSION_FILTER_ENABLED
    already controls entry-taking at the signal level, and this mode is
    meant to be triggered repeatedly by an external scheduler (cron) 24/7
    rather than deciding its own active window.
    """
    import time
    from datetime import datetime as dt

    print(f"=== Gold Paper Trader -- bounded loop mode: {loop_minutes} min total, "
          f"checking every {check_interval_minutes} min ===")
    end_time = dt.now(timezone.utc) + pd.Timedelta(minutes=loop_minutes)
    check_num = 0

    while dt.now(timezone.utc) < end_time:
        check_num += 1
        print(f"\n--- Check {check_num} (loop ends at {end_time.strftime('%H:%M UTC')}) ---")
        try:
            run_once()
        except Exception as e:
            print(f"ERROR during check: {e}")
            print("Will retry on the next check within this loop.")

        if git_commit_each_check:
            _git_commit_and_push()

        remaining = (end_time - dt.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        sleep_secs = min(check_interval_minutes * 60, remaining)
        print(f"Sleeping {sleep_secs/60:.1f} min until next check (or loop end)...")
        time.sleep(sleep_secs)

    print(f"\nBounded loop complete after {check_num} check(s). Exiting.")


def _git_commit_and_push():
    """
    Commits state/ and html/ if anything changed, and pushes. Used by
    main_bounded_loop() to persist progress after each individual check,
    rather than relying on the calling workflow to commit only once at
    the very end (which would lose all progress if the job is killed
    mid-loop). Assumes git user.name/email are already configured by the
    calling environment (the GitHub Actions workflow does this before
    invoking the script). Silently does nothing if git isn't available
    or this isn't a git repo (e.g. running locally without git) --
    prints a warning rather than crashing the whole check loop over a
    non-critical commit failure.
    """
    import subprocess
    try:
        subprocess.run(["git", "add", "state/", "html/"], check=True, capture_output=True)
        diff_check = subprocess.run(["git", "diff", "--staged", "--quiet"], capture_output=True)
        if diff_check.returncode == 0:
            print("No changes to commit this check.")
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(["git", "commit", "-m", f"Automated paper trader check: {timestamp}"],
                        check=True, capture_output=True)
        subprocess.run(["git", "push"], check=True, capture_output=True)
        print("Committed and pushed state/html updates.")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: git commit/push failed (non-fatal, continuing loop): {e}")
        if e.stderr:
            print(f"  stderr: {e.stderr.decode(errors='replace')[:500]}")
    except FileNotFoundError:
        print("WARNING: git not found on PATH -- skipping commit (non-fatal).")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if "--loop-minutes" in args:
        loop_minutes = int(args[args.index("--loop-minutes") + 1])
        check_interval_minutes = 20  # default if not specified
        if "--check-interval-minutes" in args:
            check_interval_minutes = int(args[args.index("--check-interval-minutes") + 1])
        git_commit_each_check = "--git-commit-each-check" in args
        main_bounded_loop(loop_minutes, check_interval_minutes, git_commit_each_check)
    elif "--once" in args:
        main_once()
    else:
        main_continuous()
