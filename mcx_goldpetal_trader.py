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
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# ----------------------------- CONFIG ---------------------------------
# Data source: MCX Gold Petal via Kite Connect (NOT Yahoo Finance/GC=F --
# this is a genuinely different market, currency, and contract from the
# XAUUSD work in gold_paper_trader.py).
CONFIG_PATH = r"D:\AI_Trade\Options_Snapshot\config\kite_config.env"
HISTORY_CSV = "mcx_goldpetal_stitched_history.csv"  # from mcx_stitch_history.py --
                                                       # the front-month continuous series
INTERVAL = "60minute"

EMA_FAST = 7    # IDENTICAL to gold_paper_trader.py -- same signal logic, reused as-is
EMA_SLOW = 15
EMA_STOP = 150

# ------------------------- SESSION FILTER ------------------------
# MCX Gold Petal trades 09:00-23:00 IST (up to 23:55 during US DST) --
# a genuinely different session than XAUUSD's ~24/5 forex hours. Unlike
# gold_paper_trader.py where this filter was an UNTESTED, optional
# restriction on top of 24/5 trading, here it reflects MCX's ACTUAL
# trading hours -- outside this window, the market is simply closed, not
# a discretionary choice. Kept as a config toggle for consistency with
# the reference script's structure, but should normally stay enabled for
# MCX given the market genuinely isn't open outside these hours.
SESSION_FILTER_ENABLED = True
SESSION_START_HOUR_IST = 9
SESSION_END_HOUR_IST = 23   # NOTE: extends to 23:55 during US DST -- this integer-hour
                              # check will miss the last few minutes on those days; a
                              # minor, accepted simplification for now.

GIT_COMMIT_EACH_CHECK = True

STATE_DIR = "state"
STATE_FILE = os.path.join(STATE_DIR, "paper_state.json")
TRADE_LOG_CSV = os.path.join(STATE_DIR, "trade_log.csv")
HTML_DIR = "html"
OUTPUT_HTML = os.path.join(HTML_DIR, "paper_trading_report.html")

# ------------------------- ACCOUNT / RISK CONFIG ------------------------
STARTING_CAPITAL = 100000.0   # INR, not USD -- a placeholder starting point, EDITABLE.
                                # Chosen to be large enough that 1-lot (1 gram) positions
                                # size sensibly; adjust to your actual intended capital.
RISK_PCT = 2.0
SIZING_EQUITY_CAP = 500000.0   # INR equivalent of the old $5,000 USD cap concept --
                                 # EDITABLE, not derived from any specific FX conversion,
                                 # just a placeholder ceiling matching the ORIGINAL
                                 # script's ratio (5x starting capital).

# ------------------------- PROFIT WITHDRAWAL CONFIG ------------------------
WITHDRAWAL_ENABLED = True
WITHDRAWAL_CAP_LEVEL = 500000.0   # matches SIZING_EQUITY_CAP, same withdrawal-cap
                                     # concept as gold_paper_trader.py

# ------------------------- MCX GOLD PETAL CONTRACT CONFIG ------------------------
# CONFIRMED via direct Kite instrument lookup (2026-09-12):
GOLDPETAL_LOT_SIZE_GRAMS = 1.0   # 1 lot = 1 gram, confirmed via kite.instruments()
GOLDPETAL_TICK_SIZE = 1.0        # confirmed via kite.instruments()

# APPROXIMATE, NOT CONFIRMED -- MCX SPAN margin is typically 4-7% of
# contract value and is set by the exchange DAILY, varying with
# volatility (confirmed via web research, NOT a fixed leverage ratio like
# Pepperstone's 1:200 for XAUUSD). This is a PLACEHOLDER estimate for
# backtesting purposes only -- before any real trading, verify the ACTUAL
# current margin requirement via your broker's SPAN margin calculator
# (e.g. Zerodha's margin calculator), since using a wrong margin estimate
# here could produce a materially wrong position-sizing picture.
MCX_MARGIN_FRACTION_ESTIMATE = 0.05   # 5% -- APPROXIMATE, VERIFY BEFORE LIVE USE
MARGIN_SAFETY_FRACTION = 0.20   # same conservative safety buffer concept as the
                                   # original script -- only use 20% of theoretical
                                   # max buying power on a single position

# Transaction cost: MCX brokerage + exchange charges + GST + stamp duty,
# combined into a single round-trip estimate. Real costs vary by broker
# (Zerodha charges ~₹20 flat per executed order for commodities, plus
# exchange transaction charges, GST, and SEBI charges that scale with
# turnover) -- this is a ROUGH placeholder, not a precise figure from
# your specific broker's actual charge structure.
ROUND_TRIP_COST_INR = 40.0   # APPROXIMATE placeholder (~₹20 x 2 for entry+exit) --
                                # verify against your actual Zerodha contract note
MIN_STOP_DIST_INR = 5.0   # minimum stop distance in INR to consider a trade sizeable
                             # enough to size sanely -- placeholder, not yet tuned
                             # for Gold Petal's actual typical volatility


def fetch_data_live():
    """
    Fetches recent MCX Gold Petal data LIVE from Kite Connect, for use
    during actual live/paper trading runs (main_continuous, run_once,
    etc.) -- NOT for backtesting, which instead loads the pre-fetched
    HISTORY_CSV (see run_backtest() below) to avoid hammering the API
    with thousands of historical requests every time a backtest runs.

    Fetches the CURRENT front-month contract only. Does NOT yet handle
    automatic contract rollover detection (i.e. switching to the next
    month's instrument_token when the current one is about to expire) --
    this is a known gap to address before this runs unattended across a
    contract rollover date. For now, the current front-month contract
    token must be identified and set manually (see CURRENT_CONTRACT_TOKEN
    below) and updated each time the front-month rolls.
    """
    from kiteconnect import KiteConnect
    from datetime import timedelta

    load_dotenv(CONFIG_PATH)
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise RuntimeError(f"Could not load Kite credentials from {CONFIG_PATH}")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Identify the CURRENT front-month Gold Petal contract fresh each
    # call, rather than hardcoding a token that will go stale at the next
    # rollover -- this makes rollover handling automatic for LIVE fetches
    # (the backtest's stitched history still needs re-running periodically
    # to pick up new months, since it's a static file, not a live query).
    instruments = kite.instruments(exchange="MCX")
    inst_df = pd.DataFrame(instruments)
    gold_petal = inst_df[inst_df["tradingsymbol"].str.contains("GOLDPETAL", case=False, na=False)]
    if gold_petal.empty:
        raise RuntimeError("No GOLDPETAL contracts found in Kite's current MCX instrument list.")
    gold_petal = gold_petal.copy()
    gold_petal["expiry"] = pd.to_datetime(gold_petal["expiry"])
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    valid = gold_petal[gold_petal["expiry"] >= now_ist.tz_localize(None)]
    if valid.empty:
        raise RuntimeError("No currently-valid (non-expired) GOLDPETAL contract found.")
    front_month = valid.sort_values("expiry").iloc[0]
    token = int(front_month["instrument_token"])
    symbol = front_month["tradingsymbol"]

    to_date = datetime.now()
    from_date = to_date - timedelta(days=60)  # matches the ~60d rolling window concept
                                                 # from the original XAUUSD script
    candles = kite.historical_data(
        instrument_token=token, from_date=from_date, to_date=to_date, interval=INTERVAL
    )
    if not candles:
        raise RuntimeError(f"No candles returned for {symbol} (token {token}).")

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    # Attach the contract's own tradingsymbol and expiry to every row --
    # needed for the contract-reference column and expiry-based force-close
    # logic (see process_new_bars). Constant across this fetch since it's
    # always the single current front-month contract.
    df["tradingsymbol"] = symbol
    df["expiry"] = front_month["expiry"]
    print(f"Fetched {len(df)} bars for {symbol} (front-month, token {token}, expiry {front_month['expiry'].date()})")
    return df


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
    Checks whether a bar's timestamp falls within MCX's actual trading
    hours (SESSION_START_HOUR_IST to SESSION_END_HOUR_IST), converting to
    IST first regardless of the timestamp's original timezone. Only gates
    NEW entries -- an already-open position is managed/exited regardless
    of session, matching the original script's design (risk management
    never pauses).
    """
    ts_ist = ts.tz_convert("Asia/Kolkata") if ts.tzinfo is not None else ts.tz_localize("Asia/Kolkata")
    hour = ts_ist.hour
    return SESSION_START_HOUR_IST <= hour < SESSION_END_HOUR_IST


# ------------------------- STATE MANAGEMENT ------------------------

def load_state():
    """
    State schema:
    {
      "account_balance": float,
      "open_positions": [
          {"direction": "LONG"/"SHORT", "entry_time": iso str, "entry_price": float,
           "stop_price": float, "stop_dist_inr": float, "position_grams": float,
           "position_lots": float,
           "breakeven_stop_price": float (optional, only present once armed)},
          ... (zero or more -- MULTIPLE simultaneous positions allowed, no cap,
               no same-direction restriction)
      ],
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
            "open_positions": [],
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
    # Migration: older state files (single-position era) have "open_position"
    # (dict or null) instead of "open_positions" (list). Convert on load so
    # an existing state file doesn't silently break or lose an already-open
    # position when this version is deployed.
    if "open_position" in state and "open_positions" not in state:
        old_pos = state.pop("open_position")
        state["open_positions"] = [old_pos] if old_pos is not None else []
    state.setdefault("open_positions", [])
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
        "governing_target", "target_2_revised", "stop_price", "position_grams", "position_lots", "tradingsymbol", "expiry", "gross_pnl_inr", "cost_inr",
        "net_pnl_inr", "account_balance", "withdrawal_inr", "total_withdrawn"
    ])


# ------------------------- SIZING ------------------------

def max_position_grams(gold_price_inr, sizing_equity_inr):
    """
    MCX margin is a PERCENTAGE of contract value (SPAN margin, typically
    4-7%, APPROXIMATED here at MCX_MARGIN_FRACTION_ESTIMATE -- see the
    config comment for why this is a placeholder, not a confirmed figure).
    This is a fundamentally different model from Pepperstone's fixed
    leverage ratio (1:200) used in the original XAUUSD script -- do not
    treat these two margin models as equivalent.
    """
    margin_per_gram = gold_price_inr * MCX_MARGIN_FRACTION_ESTIMATE
    theoretical_max_grams = sizing_equity_inr / margin_per_gram
    return theoretical_max_grams * MARGIN_SAFETY_FRACTION


def round_to_lot_step(grams):
    """
    Gold Petal's lot size is exactly 1 gram (confirmed via Kite instrument
    lookup) with no fractional lots below that -- so this simplifies to
    rounding DOWN to the nearest whole gram, never up (consistent with the
    original script's "never round up, never risk more than intended"
    principle).
    """
    return float(int(grams))  # floor to nearest whole gram; GOLDPETAL_LOT_SIZE_GRAMS == 1.0


def compute_position_size(account_balance_inr, stop_dist_inr, gold_price_inr):
    sizing_equity = min(account_balance_inr, SIZING_EQUITY_CAP)
    risk_inr = sizing_equity * (RISK_PCT / 100)
    position_grams = risk_inr / stop_dist_inr
    margin_cap_grams = max_position_grams(gold_price_inr, sizing_equity)
    position_grams = min(position_grams, margin_cap_grams)
    position_grams = round_to_lot_step(position_grams)
    return position_grams


# ------------------------- SIGNAL PROCESSING ------------------------

def process_new_bars(df, state):
    """
    Walks forward through any 1h bars not yet processed (per state's
    last_processed_bar), applying entry/exit logic to EVERY open position,
    one bar at a time. Only ever acts on fully CLOSED bars -- the most
    recent row from yfinance may be a still-forming current-hour bar,
    which is excluded to avoid acting on incomplete data.

    ENTRY LOGIC: UNCHANGED from the original live script -- same
    EMA_FAST/EMA_SLOW cross + pullback-touch + trend-filter conditions,
    same EMA150-based stop_price computed at entry, same
    compute_position_size() risk-based sizing. NOT the inverted-stop-fixed
    version -- this deliberately keeps the original entry/sizing logic
    exactly as currently deployed.

    EXIT LOGIC: MODIFIED. Both exit triggers (EMA150 stop hit, OR
    opposite-cross) now go through a profit check before actually closing
    the trade:
      - If the position is CURRENTLY PROFITABLE when a trigger fires,
        exit immediately (same as before).
      - If NOT profitable, do NOT exit -- instead, arm a BREAKEVEN STOP at
        the position's own entry price and keep the position open. The
        trade only closes once price returns to that breakeven level (or
        a LATER trigger re-evaluates as profitable).
      - Since the EMA150 stop is specifically designed to fire when the
        trade IS losing, this means it will almost always convert into a
        breakeven-hold rather than an immediate exit -- a deliberate,
        confirmed design choice, not an oversight.

    MULTIPLE SIMULTANEOUS POSITIONS: state["open_positions"] is a LIST.
    A new valid signal opens an ADDITIONAL position alongside any already
    open -- no limit on count, no restriction to same-direction-only.
    Each position is sized independently via the UNCHANGED
    compute_position_size() (risk-based on EMA150 stop distance), so
    total exposure grows with each additional open trade.

    Returns a list of "events" (dicts) describing what happened on each
    processed bar. Mutates state in place.
    """
    events = []

    now_utc = pd.Timestamp.now(tz="UTC")
    # PERFORMANCE: use searchsorted (binary search on the already-sorted
    # index) instead of a full boolean filter over the whole dataframe --
    # the original boolean-filter approach (df[df.index... <= now_utc])
    # re-scans the ENTIRE dataframe on every single call, which is fine
    # for a small number of calls but becomes a real bottleneck during
    # backtesting, where this function is called once per simulated check
    # (potentially thousands of times) -- confirmed via direct timing to
    # cause multi-minute runs on MCX data that should complete in seconds.
    cutoff_utc = now_utc - pd.Timedelta(hours=1)
    df_index_utc = df.index.tz_convert("UTC")
    end_pos = df_index_utc.searchsorted(cutoff_utc, side="right")

    if state["last_processed_bar"] is not None:
        last_ts_utc = pd.Timestamp(state["last_processed_bar"])
        if last_ts_utc.tzinfo is None:
            last_ts_utc = last_ts_utc.tz_localize("UTC")
        else:
            last_ts_utc = last_ts_utc.tz_convert("UTC")
        start_pos = df_index_utc.searchsorted(last_ts_utc, side="right")
    else:
        start_pos = 0

    df_closed = df.iloc[start_pos:end_pos]

    if df_closed.empty:
        return events

    for ts, row in df_closed.iterrows():
        # --- MANAGE ALL OPEN POSITIONS (each checked independently) ---
        still_open = []
        for pos in state["open_positions"]:
            d = 1 if pos["direction"] == "LONG" else -1

            # EXPIRY FORCE-CLOSE CHECK (highest priority -- overrides
            # everything else, including the "never exit at a loss"
            # principle, per explicit confirmation: the exchange settles
            # expired contracts regardless of P&L, so there is no way to
            # avoid this). If this bar's timestamp has reached or passed
            # the position's OWN contract's expiry, force-close it now at
            # this bar's real Close, labeled EXPIRY, and skip every other
            # exit check for this position this bar.
            position_expiry = pos.get("expiry")
            if position_expiry is not None and ts >= pd.Timestamp(position_expiry):
                exit_price = row["Close"]
                exit_reason = "EXPIRY"
                result = "WIN" if (exit_price - pos["entry_price"]) * d > 0 else "LOSS"

                position_grams = pos["position_grams"]
                gross_pnl = (exit_price - pos["entry_price"]) * position_grams
                if pos["direction"] == "SHORT":
                    gross_pnl = (pos["entry_price"] - exit_price) * position_grams
                cost = ROUND_TRIP_COST_INR
                net_pnl = gross_pnl - cost

                state["account_balance"] += net_pnl
                state["trade_count"] += 1

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
                    "governing_target": round(pos.get("breakeven_stop_price", pos["stop_price"]), 2),
                    "target_2_revised": round(pos["breakeven_stop_price"], 2) if pos.get("breakeven_stop_price") is not None else "",
                    "stop_price": round(pos["stop_price"], 2),
                    "position_grams": round(position_grams, 2),
                    "position_lots": round(pos["position_lots"], 2),
                    "gross_pnl_inr": round(gross_pnl, 2),
                    "cost_inr": round(cost, 2),
                    "net_pnl_inr": round(net_pnl, 2),
                    "account_balance": round(state["account_balance"], 2),
                    "withdrawal_inr": round(withdrawal_this_trade, 2),
                    "total_withdrawn": round(state["total_withdrawn"], 2),
                    "tradingsymbol": pos.get("tradingsymbol", ""),
                    "expiry": str(position_expiry)
                }
                append_trade_log(trade_record)
                events.append({"type": "EXIT", "reason": exit_reason,
                                "result": result, **trade_record})
                continue  # position force-closed -- skip all other checks for it this bar

            # BREAKEVEN STOP CHECK (only active once armed): if a prior
            # trigger found THIS position unprofitable, a stop was set at
            # ITS entry price -- waiting for price to RECOVER back to that
            # level before closing. Checked first, using the same
            # conservative same-bar logic (adverse wick checked before
            # anything else this bar).
            #
            # FIX: for a LONG that's underwater (price below entry), we are
            # waiting for price to RISE back up to breakeven -- so the
            # correct check is whether the bar's HIGH reached UP to that
            # level (High >= breakeven_stop), not the Low. The previous
            # version checked Low <= breakeven_stop for LONGs, which is
            # satisfied by almost ANY bar while the position is underwater
            # (since Low is naturally below a breakeven level price hasn't
            # reached yet), causing an exit to fire even when price never
            # actually recovered (confirmed directly: a bar with
            # High=15371, entry/breakeven=15378 triggered an exit despite
            # price never reaching the breakeven level at all). Symmetric
            # fix applies to SHORT (checks Low <= breakeven_stop, price
            # falling back down to entry from above).
            breakeven_stop = pos.get("breakeven_stop_price")
            hit_breakeven = breakeven_stop is not None and (
                (d == 1 and row["High"] >= breakeven_stop) or
                (d == -1 and row["Low"] <= breakeven_stop)
            )

            exit_price = None
            exit_reason = None

            # FIX (per explicit design decision): the LEVEL (breakeven price,
            # or EMA150 stop_price) is used ONLY to decide WHETHER to exit --
            # it is a TARGET/TRIGGER, not necessarily a price the market
            # actually traded at. Once a trigger fires, the trade is recorded
            # as filling at THIS BAR'S ACTUAL CLOSE (a real, tradeable
            # price), not the target level itself. This directly addresses a
            # confirmed issue: checking 24 of 25 real STOP exits in one
            # backtest run showed the market never actually traded at the
            # target level in 96% of cases (the target was often far above/
            # below anywhere price genuinely reached), meaning P&L computed
            # from the target alone could show a "profit" the market never
            # actually offered. Using the bar's real Close for P&L, while
            # still using the target/level purely to decide WHEN to exit,
            # keeps the strategy's trigger logic unchanged while ensuring
            # every recorded P&L reflects a genuine, tradeable price.
            if hit_breakeven:
                # FIX (per explicit follow-up correction, same principle as
                # the STOP/OPPOSITE_CROSS fix above): a breakeven TOUCH
                # (intrabar wick reaching the level) does NOT guarantee the
                # bar's actual CLOSE is also at/better than breakeven -- a
                # bar can wick down to touch breakeven and then close
                # somewhere else entirely. Confirmed directly: a SHORT with
                # breakeven=15325 had a bar with Low=15321 (touching
                # breakeven) but Close=15332 (still a real loss for the
                # SHORT) -- exiting unconditionally on the touch recorded a
                # real loss at a moment when the position hadn't actually
                # recovered. Now: only ACTUALLY exit if the real Close
                # confirms breakeven-or-better; otherwise keep holding
                # (breakeven stays armed for the next bar).
                real_price_now = row["Close"]
                close_confirms_breakeven = (real_price_now - pos["entry_price"]) * d >= 0
                if close_confirms_breakeven:
                    exit_price = real_price_now
                    exit_reason = "BREAKEVEN_STOP"
                # else: touched intrabar but Close didn't confirm -- stay
                # open, breakeven remains armed (pos["breakeven_stop_price"]
                # is already set, no change needed).
            else:
                # ORIGINAL exit triggers, UNCHANGED as TRIGGERS: EMA150
                # stop_price hit, OR opposite-cross. Re-evaluated on every
                # bar for every still-open position independently.
                stop_price = pos["stop_price"]
                hit_stop = (d == 1 and row["Low"] <= stop_price) or \
                           (d == -1 and row["High"] >= stop_price)
                opposite_cross = row["CROSS"] == (-d)

                if hit_stop or opposite_cross:
                    # FIX (per explicit follow-up correction): the decision
                    # of WHETHER TO EXIT NOW vs. HOLD FOR BREAKEVEN must be
                    # based on the REAL price (this bar's actual Close), not
                    # the target/level. The earlier version of this fix
                    # still checked profitability against the target level,
                    # which meant a trade could be judged "profitable" by
                    # the target and exit immediately, even when the REAL
                    # price at that moment was actually a loss (confirmed:
                    # a trade with target=15466.21 > entry showed
                    # "profitable" and exited immediately, but the bar's
                    # real Close (15332) was BELOW entry -- a genuine loss
                    # that should instead have armed the breakeven hold).
                    real_price_now = row["Close"]
                    is_really_profitable = (real_price_now - pos["entry_price"]) * d > 0
                    if is_really_profitable:
                        exit_price = real_price_now
                        exit_reason = "STOP" if hit_stop else "OPPOSITE_CROSS"
                    else:
                        # NOT profitable at the REAL current price -- do NOT
                        # exit. Arm/re-arm the breakeven stop at THIS
                        # position's entry price instead, exactly as
                        # originally requested: "before exiting the trade
                        # with loss, set target to breakeven."
                        pos["breakeven_stop_price"] = pos["entry_price"]
                        # Write an updated OPEN record so Target-2 (revised
                        # target) becomes visible in the report AS SOON AS
                        # breakeven is armed, not only once the trade
                        # eventually closes -- upserts the existing OPEN
                        # row by entry_time (same mechanism append_trade_log
                        # always uses).
                        append_trade_log({
                            "entry_time": pos["entry_time"], "status": "OPEN",
                            "target_2_revised": round(pos["breakeven_stop_price"], 2)
                        })

            if exit_price is not None:
                result = "WIN" if (exit_price - pos["entry_price"]) * d > 0 else "LOSS"

                position_grams = pos["position_grams"]
                gross_pnl = (exit_price - pos["entry_price"]) * position_grams
                if pos["direction"] == "SHORT":
                    gross_pnl = (pos["entry_price"] - exit_price) * position_grams
                cost = ROUND_TRIP_COST_INR
                net_pnl = gross_pnl - cost

                state["account_balance"] += net_pnl
                state["trade_count"] += 1

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
                    # governing_target = whichever level ACTUALLY triggered
                    # this specific exit: the breakeven level if that's what
                    # fired, otherwise the original EMA150 stop set at entry.
                    # This is the TARGET that was set to decide WHEN to exit
                    # -- exit_price is the REAL price the trade actually
                    # closed at, which may differ from this target (that's
                    # the whole point of the earlier fix -- see comments in
                    # the exit-logic block above).
                    "governing_target": round(breakeven_stop, 2) if exit_reason == "BREAKEVEN_STOP" else round(pos["stop_price"], 2),
                    "target_2_revised": round(breakeven_stop, 2) if breakeven_stop is not None else "",
                    "stop_price": round(pos["stop_price"], 2),
                    "position_grams": round(position_grams, 2),
                    "position_lots": round(pos["position_lots"], 2),
                    "gross_pnl_inr": round(gross_pnl, 2),
                    "cost_inr": round(cost, 2),
                    "net_pnl_inr": round(net_pnl, 2),
                    "account_balance": round(state["account_balance"], 2),
                    "withdrawal_inr": round(withdrawal_this_trade, 2),
                    "total_withdrawn": round(state["total_withdrawn"], 2)
                }
                append_trade_log(trade_record)
                events.append({"type": "EXIT", "reason": exit_reason,
                                "result": result, **trade_record})
                # This position closed -- do NOT add it back to still_open.
            else:
                still_open.append(pos)

        state["open_positions"] = still_open

        # --- LOOK FOR A NEW SETUP/ENTRY -- UNCHANGED logic, but now
        # regardless of how many positions are ALREADY open. ---
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
                    events.append({"type": "SIGNAL_SKIPPED", "reason": "outside_session_window",
                                    "entry_time": str(ts)})
                    state["setup_direction"] = 0
                else:
                    entry_price = row["Close"]
                    trade_dir = state["setup_direction"]
                    initial_stop = row["EMA_STOP"]
                    stop_dist = abs(entry_price - initial_stop)

                    if stop_dist < MIN_STOP_DIST_INR:
                        events.append({"type": "SIGNAL_SKIPPED", "reason": "stop_too_tight",
                                        "entry_time": str(ts), "stop_dist_inr": round(stop_dist, 4)})
                        state["setup_direction"] = 0
                    else:
                        position_grams = compute_position_size(state["account_balance"], stop_dist, entry_price)
                        if position_grams <= 0:
                            events.append({"type": "SIGNAL_SKIPPED", "reason": "sizes_to_zero_lots",
                                            "entry_time": str(ts)})
                            state["setup_direction"] = 0
                        else:
                            new_position = {
                                "direction": "LONG" if trade_dir == 1 else "SHORT",
                                "entry_time": str(ts), "entry_price": entry_price,
                                "stop_price": initial_stop, "stop_dist_inr": stop_dist,
                                "position_grams": position_grams,
                                "position_lots": position_grams / GOLDPETAL_LOT_SIZE_GRAMS,
                                "tradingsymbol": row.get("tradingsymbol", ""),
                                "expiry": str(row.get("expiry", "")) if row.get("expiry") is not None else None
                            }
                            state["open_positions"].append(new_position)
                            state["setup_direction"] = 0

                            trade_record = {
                                "entry_time": str(ts), "exit_time": "",
                                "direction": "LONG" if trade_dir == 1 else "SHORT",
                                "status": "OPEN",
                                "entry_price": round(entry_price, 2), "exit_price": "",
                                "governing_target": round(initial_stop, 2),
                                "target_2_revised": "",
                                "stop_price": round(initial_stop, 2),
                                "position_grams": round(position_grams, 2),
                                "position_lots": round(position_grams / GOLDPETAL_LOT_SIZE_GRAMS, 2),
                                "tradingsymbol": row.get("tradingsymbol", ""),
                                "expiry": str(row.get("expiry", "")) if row.get("expiry") is not None else "",
                                "gross_pnl_inr": "", "cost_inr": "", "net_pnl_inr": "",
                                "account_balance": round(state["account_balance"], 2),
                                "withdrawal_inr": 0.0, "total_withdrawn": round(state["total_withdrawn"], 2)
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

    open_positions = state["open_positions"]
    open_pos_block = ""
    if open_positions:
        open_pos_block = f"""
    <div class="section-title">Open Positions ({len(open_positions)})</div>"""
        for open_pos in open_positions:
            d_color = "#16a34a" if open_pos["direction"] == "LONG" else "#dc2626"
            d = 1 if open_pos["direction"] == "LONG" else -1
            breakeven = open_pos.get("breakeven_stop_price")
            breakeven_note = (f"<div class=\"stat-card\"><div class=\"stat-label\">Breakeven Stop</div>"
                               f"<div class=\"stat-value\" style=\"color:#f97316\">Rs.{breakeven:.2f} (armed)</div></div>"
                               if breakeven is not None else
                               "<div class=\"stat-card\"><div class=\"stat-label\">Breakeven Stop</div>"
                               "<div class=\"stat-value\" style=\"color:#6b7280\">Not armed</div></div>")

            # Last known price and unrealised P&L
            sym = open_pos.get("tradingsymbol", "")
            last_price = state.get("last_price_by_symbol", {}).get(sym)
            if last_price is not None:
                upnl = (last_price - open_pos["entry_price"]) * d * open_pos["position_grams"]
                upnl_color = "#3fb950" if upnl >= 0 else "#f85149"
                upnl_sign = "+" if upnl >= 0 else ""
                last_price_card = (f"<div class=\"stat-card\"><div class=\"stat-label\">Last Price</div>"
                                   f"<div class=\"stat-value\">Rs.{last_price:.2f}</div></div>")
                upnl_card = (f"<div class=\"stat-card\" style=\"border-color:{upnl_color}\">"
                             f"<div class=\"stat-label\">If Closed Now</div>"
                             f"<div class=\"stat-value\" style=\"color:{upnl_color};font-size:18px\">"
                             f"{upnl_sign}Rs.{upnl:,.0f}</div></div>")
            else:
                last_price_card = ("<div class=\"stat-card\"><div class=\"stat-label\">Last Price</div>"
                                   "<div class=\"stat-value\" style=\"color:#6b7280\">—</div></div>")
                upnl_card = ("<div class=\"stat-card\"><div class=\"stat-label\">If Closed Now</div>"
                             "<div class=\"stat-value\" style=\"color:#6b7280\">—</div></div>")

            open_pos_block += f"""
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-label">Direction</div><div class="stat-value" style="color:{d_color}">{open_pos['direction']}</div></div>
        <div class="stat-card"><div class="stat-label">Entry Time</div><div class="stat-value" style="font-size:14px">{open_pos['entry_time']}</div></div>
        <div class="stat-card"><div class="stat-label">Entry Price</div><div class="stat-value">Rs.{open_pos['entry_price']:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">EMA Stop Price</div><div class="stat-value">Rs.{open_pos['stop_price']:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Size</div><div class="stat-value">{open_pos['position_grams']:.2f}g ({open_pos['position_lots']:.2f} lot)</div></div>
        {breakeven_note}
        {last_price_card}
        {upnl_card}
    </div>"""
    else:
        open_pos_block = """
    <div class="section-title">Open Positions (0)</div>
    <div class="verdict" style="border-left-color:#6b7280;">No open positions -- scanning for the next setup.</div>"""

    events_block = ""
    if events:
        event_items = ""
        for e in events:
            if e["type"] == "ENTRY":
                event_items += f"<li>ENTRY: {e['direction']} at Rs.{e['entry_price']:.2f}, bar {e['entry_time']}</li>"
            elif e["type"] == "EXIT":
                color = "#16a34a" if e["result"] == "WIN" else "#dc2626"
                event_items += (f"<li style='color:{color}'>EXIT ({e['reason']}, {e['result']}): "
                                 f"{e['direction']} closed at Rs.{e['exit_price']:.2f}, net Rs.{e['net_pnl_inr']:.2f}, "
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
                            ("#16a34a" if str(t.get("net_pnl_inr", "")) not in ("", "nan") and float(t["net_pnl_inr"]) > 0 else "#dc2626")
            dir_color = "#16a34a" if t["direction"] == "LONG" else "#dc2626"
            exit_price_str = f"Rs.{float(t['exit_price']):.2f}" if str(t["exit_price"]) not in ("", "nan") else "-"
            target1_val = t.get("governing_target", "")
            target1_str = f"Rs.{float(target1_val):.2f}" if str(target1_val) not in ("", "nan") else "-"
            target2_val = t.get("target_2_revised", "")
            target2_str = f"Rs.{float(target2_val):.2f}" if str(target2_val) not in ("", "nan") else "-"
            net_pnl_str = f"Rs.{float(t['net_pnl_inr']):.2f}" if str(t["net_pnl_inr"]) not in ("", "nan") else "-"
            withdrawal_val = t.get("withdrawal_inr", 0.0)
            withdrawal_str = (f"Rs.{float(withdrawal_val):.2f}"
                               if str(withdrawal_val) not in ("", "nan") and float(withdrawal_val) > 0 else "-")
            symbol_val = t.get("tradingsymbol", "") or "-"
            expiry_val = t.get("expiry", "")
            expiry_str = str(expiry_val)[:10] if str(expiry_val) not in ("", "nan", "None") else "-"
            trade_rows += f"""
        <tr>
            <td>{t['entry_time']}</td>
            <td style="color:{dir_color};font-weight:600">{t['direction']}</td>
            <td style="color:{status_color};font-weight:600">{t['status']}</td>
            <td>Rs.{float(t['entry_price']):.2f}</td>
            <td>{exit_price_str}</td>
            <td style="color:#9ca3af;font-size:12px">{target1_str}</td>
            <td style="color:#eab308;font-size:12px">{target2_str}</td>
            <td>{t['position_grams']:.2f}g ({t['position_lots']:.2f} lot)</td>
            <td style="color:{status_color};font-weight:600">{net_pnl_str}</td>
            <td>Rs.{float(t['account_balance']):.2f}</td>
            <td style="color:#f59e0b">{withdrawal_str}</td>
            <td style="font-size:12px;color:#9ca3af">{symbol_val}</td>
            <td style="font-size:12px;color:#9ca3af">{expiry_str}</td>
        </tr>"""

    closed = trade_log_df[trade_log_df["status"] == "CLOSED"] if not trade_log_df.empty else pd.DataFrame()
    wins = (closed["net_pnl_inr"].astype(float) > 0).sum() if not closed.empty else 0
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
    <div class="subtitle">EMA{EMA_FAST}/{EMA_SLOW} cross + pullback, EMA{EMA_STOP} fixed stop, {RISK_PCT:.0f}% risk, {MCX_MARGIN_FRACTION_ESTIMATE*100:.0f}% margin (approx) @ {MARGIN_SAFETY_FRACTION*100:.0f}% safety cap</div>
    <div class="subtitle">{'Session filter ON: new entries only ' + str(SESSION_START_HOUR_IST) + '-' + str(SESSION_END_HOUR_IST) + ' IST (MCX actual trading hours)' if SESSION_FILTER_ENABLED else 'Session filter OFF: 24/5, matches validated backtest exactly'}</div>
    <div class="subtitle" style="color:#f59e0b; font-weight:600;">Config: EMA_FAST={EMA_FAST}, EMA_SLOW={EMA_SLOW}, EMA_STOP={EMA_STOP}, STARTING_CAPITAL=Rs.{STARTING_CAPITAL:.0f}, SIZING_EQUITY_CAP=Rs.{SIZING_EQUITY_CAP:,.0f} -- check this matches the backtest you're comparing against</div>
    <div class="subtitle">Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>

    <div class="warning">
        ⚠ This is a PAPER TRACKER, not an auto-trader. It tells you what to do based on completed
        1h candles -- place/close the actual trade yourself on your Pepperstone cTrader demo account.
        Position sizing is capped for calculation purposes at Rs.{SIZING_EQUITY_CAP:,.0f} equity -- growth
        continues past that, but position size stops increasing further once real balance exceeds it.
    </div>

    <div class="section-title">Account Status</div>
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-label">Account Balance</div><div class="stat-value">Rs.{state['account_balance']:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Total Withdrawn</div><div class="stat-value">Rs.{total_withdrawn:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Total Wealth Return</div><div class="stat-value">{total_wealth_return_pct:.2f}%</div></div>
        <div class="stat-card"><div class="stat-label">Total Trades</div><div class="stat-value">{state['trade_count']}</div></div>
        <div class="stat-card"><div class="stat-label">Win Rate (closed trades)</div><div class="stat-value">{win_rate}%</div></div>
    </div>

    {open_pos_block}

    {events_block}

    <div class="section-title">Full Trade History</div>
    <div style="background:#3f2d0f; border:1px solid #d97706; color:#fbbf24; padding:12px 16px;
                border-radius:8px; font-size:13px; margin-bottom:12px;">
        &#9888; <strong>Contract/Expiry data limitation:</strong> every trade below shows
        <strong>GOLDPETAL26SEPFUT</strong> regardless of its actual entry month. This is
        because Kite Connect does not expose already-expired contracts' historical data
        (confirmed via direct testing) -- so this backtest uses ONE contract's real price
        history throughout (extending back to its own true listing start), rather than
        genuinely rolling between each month's actual front-month contract the way a real
        trader would have. Prices themselves are real and were genuinely traded on the
        September contract at each shown timestamp -- but for entries before ~August 2026,
        the September contract may NOT have been the realistic front-month choice at that
        time. Treat this as a single-contract historical smoke test, not a realistic
        month-by-month rollover simulation. Genuine rolling-contract data can only be
        captured going forward, in real time, as each new month's contract becomes current.
    </div>
    <table>
        <tr><th>Entry Time</th><th>Direction</th><th>Status</th><th>Entry</th><th>Exit</th><th>Target-1 (Original)</th><th>Target-2 (Revised)</th><th>Size</th><th>Net P&L</th><th>Account Balance</th><th>Withdrawal</th><th>Contract</th><th>Expiry</th></tr>
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
          f"STARTING_CAPITAL=Rs.{STARTING_CAPITAL:.0f}, RISK_PCT={RISK_PCT:.0f}%, "
          f"SIZING_EQUITY_CAP=Rs.{SIZING_EQUITY_CAP:,.0f}")
    if SESSION_FILTER_ENABLED:
        print(f"Session filter: ON -- new entries only {SESSION_START_HOUR_IST}:00-{SESSION_END_HOUR_IST}:00 IST "
              f"(untested vs the validated backtest, which ran 24/5)")
    else:
        print("Session filter: OFF -- 24/5, matches validated backtest exactly")
    state = load_state()
    open_positions_summary = ", ".join(p["direction"] for p in state["open_positions"])
    print(f"Loaded state: balance=Rs.{state['account_balance']:.2f}, "
          f"open_positions=[{open_positions_summary}] ({len(state['open_positions'])} open), "
          f"trades so far={state['trade_count']}")

    df = fetch_data_live()
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
                print(f"  -> ENTER {e['direction']} at Rs.{e['entry_price']:.2f}, "
                      f"size {e['position_grams']:.2f}g ({e['position_lots']:.2f} lot), "
                      f"stop Rs.{e['stop_price']:.2f}, bar {e['entry_time']}")
            elif e["type"] == "EXIT":
                print(f"  -> EXIT ({e['reason']}, {e['result']}): {e['direction']} closed at "
                      f"Rs.{e['exit_price']:.2f}, net Rs.{e['net_pnl_inr']:.2f}, balance now "
                      f"Rs.{e['account_balance']:.2f}, bar {e['exit_time']}")
            elif e["type"] == "SIGNAL_SKIPPED":
                print(f"  -> Signal skipped ({e['reason']}) at bar {e['entry_time']}")
    else:
        print("No new completed bars since last check -- nothing to process.")

    open_positions_summary2 = ", ".join(p["direction"] for p in state["open_positions"])
    print(f"Current balance: Rs.{state['account_balance']:.2f} | "
          f"Open positions: [{open_positions_summary2}] ({len(state['open_positions'])} open)")

    trade_log_df = read_trade_log()
    html = build_html_report(state, events, trade_log_df)
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report updated: {OUTPUT_HTML}")


def _seconds_until_next_window_open():
    """
    When outside the active IST window, computes how long to sleep before
    the window next opens (today if it hasn't started yet, tomorrow if
    today's window has already closed).
    """
    now = datetime.now(timezone.utc)
    today_open = now.replace(hour=SESSION_START_HOUR_IST, minute=0, second=0, microsecond=0)
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
    Upstox ratchet bot's intraday loop.

    When SESSION_FILTER_ENABLED is True, the loop is only ACTIVE (checking)
    during the configured IST session window; outside that window it
    sleeps until the window next opens rather than exiting, so it can be
    left running unattended. An open position is still monitored for
    exits even though new entries are session-gated -- risk management
    never pauses.

    When SESSION_FILTER_ENABLED is False (the current setting), the loop
    checks continuously 24/7 -- consistent with entries also being allowed
    at any hour (per process_new_bars()'s own SESSION_FILTER_ENABLED
    check). Previously this outer loop ALWAYS gated on the IST window
    regardless of SESSION_FILTER_ENABLED, which silently limited checking
    to the configured IST window even after entries were opened up to 24/7 -- fixed here
    so both layers agree.

    POLL_INTERVAL_SECONDS controls how often it checks for a newly closed
    1h candle while active.

    If GIT_COMMIT_EACH_CHECK is True, commits+pushes state/html after every
    check (not just at the very end) -- keeps the GitHub repo's report
    current as a viewable source of truth, without needing to SSH into
    the host running this loop. Requires git to be configured with a
    remote and credentials that can push without interactive auth (e.g.
    a deploy key or credential helper) -- see _git_commit_and_push()'s
    docstring for the non-fatal failure behavior if that's not set up.

    Requires a host that stays continuously awake (a paid always-on task,
    a systemd service on a real VM, or a machine that never sleeps). NOT
    suitable for a free-tier daily scheduled task or a laptop that sleeps
    -- use main_once() for those.
    """
    POLL_INTERVAL_SECONDS = 20 * 60  # 20 minutes, within the requested 15-30 min range

    print("=== Gold Paper Trader -- continuous mode ===")
    if SESSION_FILTER_ENABLED:
        print(f"Active window: {SESSION_START_HOUR_IST}:00-{SESSION_END_HOUR_IST}:00 IST, "
              f"polling every {POLL_INTERVAL_SECONDS // 60} minutes while active.")
    else:
        print(f"Session filter OFF -- checking 24/7, every {POLL_INTERVAL_SECONDS // 60} minutes.")
    if GIT_COMMIT_EACH_CHECK:
        print("Git commit+push after each check: ENABLED.")
    print("Leave this window running. Press Ctrl+C to stop.\n")

    import time
    while True:
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        in_active_window = (not SESSION_FILTER_ENABLED) or \
                            (SESSION_START_HOUR_IST <= current_hour < SESSION_END_HOUR_IST)

        if in_active_window:
            try:
                run_once()
            except Exception as e:
                # Never let one failed check (e.g. a transient data-fetch
                # error) kill the whole session -- log it and keep going,
                # since an open position still needs to be monitored on
                # the next cycle.
                print(f"ERROR during check: {e}")
                print("Will retry on the next cycle.")

            if GIT_COMMIT_EACH_CHECK:
                _git_commit_and_push()

            print(f"Sleeping {POLL_INTERVAL_SECONDS // 60} minutes until next check...")
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            sleep_secs = _seconds_until_next_window_open()
            wake_time = now + pd.Timedelta(seconds=sleep_secs)
            print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Outside active window "
                  f"({SESSION_START_HOUR_IST}:00-{SESSION_END_HOUR_IST}:00 IST). "
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

    Unlike main_continuous(), this does NOT gate on SESSION_START_HOUR_IST/
    SESSION_END_HOUR_IST sleep-until-window logic -- it runs for its
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


# ===================================================================
# ROLLING-WINDOW BACKTEST MODE -- run with: python3 gold_paper_trader_livebe.py --backtest
# Tests THIS file's own logic (unmodified live entry/EMA150-stop logic,
# PLUS the breakeven-instead-of-immediate-exit rule and multiple
# simultaneous positions) against real historical data, using a rolling
# 60-day window at each simulated check -- genuinely matching what a
# live deployment of THIS SAME FILE would see and do.
# ===================================================================

ROLLING_WINDOW_DAYS = 60        # MUST match this file's own live-fetch window concept
CHECK_INTERVAL_HOURS = 1
BACKTEST_OUTPUT_DIR = "rolling_output_mcx"
BACKTEST_OUTPUT_HTML = os.path.join(BACKTEST_OUTPUT_DIR, "rolling_window_report.html")
BACKTEST_OUTPUT_CSV = os.path.join(BACKTEST_OUTPUT_DIR, "rolling_window_trade_log.csv")


def fetch_master_data():
    """
    UNLIKE the original XAUUSD script (which fetched fresh from Yahoo for
    every backtest run), this loads the PRE-FETCHED, already-stitched
    front-month continuous series produced by mcx_stitch_history.py --
    see HISTORY_CSV config above. Re-run mcx_stitch_history.py separately
    whenever you want to refresh this file with more recent data (e.g.
    after a new month has passed and there's more real history available).
    This script does NOT re-fetch from Kite itself during a backtest, to
    avoid hammering the API with a full historical re-fetch every single
    backtest run.
    """
    if not os.path.exists(HISTORY_CSV):
        raise RuntimeError(
            f"{HISTORY_CSV} not found. Run mcx_stitch_history.py first to "
            f"produce the front-month continuous price series this backtest "
            f"needs."
        )
    df = pd.read_csv(HISTORY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    # Keep tradingsymbol and expiry (both already present in the stitched
    # CSV from mcx_stitch_history.py) -- needed for the contract-reference
    # column and the expiry-based force-close logic below. Previously these
    # were dropped here, silently discarding contract metadata the rest of
    # the pipeline needs.
    keep_cols = ["Open", "High", "Low", "Close"]
    if "tradingsymbol" in df.columns:
        keep_cols.append("tradingsymbol")
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"])
        keep_cols.append("expiry")
    df = df[keep_cols].dropna(subset=["Open", "High", "Low", "Close"])
    print(f"Loaded {len(df)} bars from {HISTORY_CSV}")
    return df


def build_backtest_trade_log_df(all_events):
    """
    Reconstructs the trade_log_df shape build_html_report() expects,
    with an OPEN row per entry updated in place to CLOSED on exit
    (matching append_trade_log()'s real upsert behavior), rather than one
    row per event. Includes stop_price (the EMA150 level, as this
    version's real trade records do) plus exit_reason for diagnostics.
    """
    log_df = pd.DataFrame(columns=[
        "entry_time", "exit_time", "direction", "status", "entry_price", "exit_price",
        "governing_target", "target_2_revised", "stop_price", "position_grams", "position_lots", "tradingsymbol", "expiry", "gross_pnl_inr", "cost_inr",
        "net_pnl_inr", "account_balance", "withdrawal_inr", "total_withdrawn", "exit_reason"
    ])
    for e in all_events:
        if e["type"] == "ENTRY":
            row = {k: e[k] for k in log_df.columns if k in e}
            row["exit_reason"] = ""
            log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
        elif e["type"] == "EXIT":
            match = log_df["entry_time"].astype(str) == str(e["entry_time"])
            exit_reason_value = e.get("reason", "")
            if match.any():
                idx = log_df.index[match][0]
                for k in log_df.columns:
                    if k in e:
                        log_df.loc[idx, k] = e[k]
                log_df.loc[idx, "exit_reason"] = exit_reason_value
            else:
                row = {k: e[k] for k in log_df.columns if k in e}
                row["exit_reason"] = exit_reason_value
                log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    return log_df


def _inject_exit_reason_column(html, trade_log_df):
    """
    Adds an 'Exit Reason' column (STOP / OPPOSITE_CROSS / BREAKEVEN_STOP)
    to the trade history table in the generated HTML, without modifying
    build_html_report() itself. Matched by the SAME row order
    build_html_report() uses internally (entry_time descending).
    """
    import re

    html = html.replace(
        "<th>Entry Time</th><th>Direction</th><th>Status</th>",
        "<th>Entry Time</th><th>Direction</th><th>Status</th><th>Exit Reason</th>"
    )

    if trade_log_df.empty:
        return html

    sorted_log = trade_log_df.sort_values("entry_time", ascending=False)
    reason_labels = []
    for _, t in sorted_log.iterrows():
        reason = t.get("exit_reason", "")
        # FIX: label now reflects the trade's REAL, ACTUAL outcome
        # (net_pnl_inr), not just "which trigger fired" -- since the fix
        # that makes exit_price always the bar's real Close means a
        # STOP/OPPOSITE_CROSS trigger judged "profitable" against the
        # TARGET level can still turn out to be a real loss once priced at
        # the actual traded Close (this was confirmed happening: e.g. a
        # trade triggered because the target level looked profitable, but
        # the real fill came in at an actual loss). The old label
        # unconditionally said "(profitable)" for every STOP exit, which
        # became actively misleading once exit_price stopped being the
        # target level itself.
        net_pnl = t.get("net_pnl_inr", None)
        is_real_win = (str(net_pnl) not in ("", "nan", "None")) and float(net_pnl) > 0
        outcome_word = "profit" if is_real_win else "loss"
        outcome_color = "#16a34a" if is_real_win else "#dc2626"
        if reason == "STOP":
            reason_labels.append(f'<td style="color:#f97316;font-size:12px">EMA150 stop '
                                  f'(<span style="color:{outcome_color}">{outcome_word}</span>)</td>')
        elif reason == "BREAKEVEN_STOP":
            reason_labels.append(f'<td style="color:#eab308;font-size:12px">Breakeven stop '
                                  f'(<span style="color:{outcome_color}">{outcome_word}</span>)</td>')
        elif reason == "OPPOSITE_CROSS":
            reason_labels.append(f'<td style="color:#3b82f6;font-size:12px">Opposite cross '
                                  f'(<span style="color:{outcome_color}">{outcome_word}</span>)</td>')
        elif reason == "EXPIRY":
            # Force-closed due to contract expiry -- regardless of outcome,
            # this is worth flagging distinctly (not a strategy decision,
            # a hard exchange-imposed deadline). Per explicit confirmation:
            # this overrides the "never exit at a loss" rule, since the
            # exchange settles expired contracts regardless of P&L.
            reason_labels.append(f'<td style="color:#a855f7;font-size:12px">Contract expiry '
                                  f'(<span style="color:{outcome_color}">{outcome_word}</span>)</td>')
        else:
            reason_labels.append('<td style="color:#9ca3af;font-size:12px">-</td>')

    row_pattern = re.compile(r'(<tr>\s*<td>.*?</td>\s*<td[^>]*>.*?</td>\s*<td[^>]*>.*?</td>\s*)(<td>Rs\.)', re.DOTALL)
    row_index = [0]
    def _insert_reason(match):
        idx = row_index[0]
        row_index[0] += 1
        if idx < len(reason_labels):
            return match.group(1) + reason_labels[idx] + match.group(2)
        return match.group(0)
    html = row_pattern.sub(_insert_reason, html)
    return html


def run_backtest():
    import sys
    print("=" * 70)
    print("ROLLING-WINDOW BACKTEST (genuinely matches live VM's 60-day rolling context)")
    print("=" * 70)
    print(f"Config: EMA_FAST={EMA_FAST}, EMA_SLOW={EMA_SLOW}, EMA_STOP={EMA_STOP}")
    print(f"ROLLING_WINDOW_DAYS={ROLLING_WINDOW_DAYS} (matches this file's own PERIOD exactly)")
    print(f"NOTE: this file has the UNMODIFIED live entry/EMA150-stop logic, PLUS: "
          f"(1) multiple simultaneous positions allowed, and (2) BOTH exit triggers "
          f"(EMA150 stop hit, opposite-cross) now check profit first -- if profitable, "
          f"exit immediately; if not, HOLD and arm a breakeven stop at entry price instead "
          f"of exiting at a loss. Since the EMA150 stop is designed to fire when losing, "
          f"expect it to almost always convert into a breakeven-hold rather than an "
          f"immediate exit.")
    print("=" * 70)

    master_df = fetch_master_data()
    span_days = (master_df.index.max() - master_df.index.min()).days
    print(f"\nMaster dataset: {len(master_df)} bars, {master_df.index.min()} to "
          f"{master_df.index.max()} (~{span_days} days)")
    if span_days < 365:
        print(f"NOTE: this backtest covers only ~{span_days} days of real MCX history -- "
              f"a genuine multi-year validation (like the 700+ trade, 2+ year XAUUSD "
              f"backtest) is NOT possible yet, since Gold Petal contracts only became "
              f"accessible via Kite starting from their current listing dates. Treat "
              f"this run as a MECHANICAL smoke test (confirming entries/exits/sizing "
              f"work correctly on real MCX data), not a statistically meaningful "
              f"validation of the strategy on this instrument. Re-run "
              f"mcx_stitch_history.py periodically to extend this history as more "
              f"real months become available.")

    master_df = master_df.sort_index()
    master_df = add_indicators(master_df.copy())

    sim_start = master_df.index.min() + pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    sim_end = master_df.index.max()
    print(f"Simulating checks from {sim_start} to {sim_end} "
          f"(every {CHECK_INTERVAL_HOURS}h simulated 'now')")

    state = {
        "account_balance": STARTING_CAPITAL,
        "open_positions": [],
        "setup_direction": 0,
        "last_processed_bar": None,
        "trade_count": 0,
        "total_withdrawn": 0.0,
        "initialized": True
    }

    all_events = []
    current_sim_time = sim_start
    check_count = 0
    total_checks = int((sim_end - sim_start).total_seconds() / 3600 / CHECK_INTERVAL_HOURS) + 1

    # CRITICAL: monkey-patch pd.Timestamp.now so the REAL, unmodified
    # process_new_bars() can be called directly during simulation -- see
    # the no-stop version's development notes for why this approach
    # (rather than a hand-copied duplicate) is used: a hand-copy risks
    # silently drifting from the actual function's real behavior.
    import unittest.mock as mock
    real_timestamp_now = pd.Timestamp.now

    class _SimulatedNow:
        current = None
        @classmethod
        def now(cls, tz=None):
            if cls.current is None:
                return real_timestamp_now(tz=tz)
            return cls.current.tz_convert(tz) if tz else cls.current

    def _noop_append_trade_log(row_dict):
        pass

    with mock.patch.object(pd.Timestamp, "now", side_effect=_SimulatedNow.now), \
         mock.patch.object(sys.modules[__name__], "append_trade_log", _noop_append_trade_log):

        while current_sim_time <= sim_end:
            check_count += 1
            if check_count % 2000 == 0:
                print(f"  ...simulated check {check_count}/{total_checks} "
                      f"({current_sim_time.strftime('%Y-%m-%d %H:%M')})")

            _SimulatedNow.current = current_sim_time.tz_localize("UTC") if current_sim_time.tzinfo is None \
                                     else current_sim_time.tz_convert("UTC")
            events = process_new_bars(master_df, state)
            all_events.extend(events)

            current_sim_time += pd.Timedelta(hours=CHECK_INTERVAL_HOURS)

    entries = [e for e in all_events if e["type"] == "ENTRY"]
    exits = [e for e in all_events if e["type"] == "EXIT"]
    skipped = [e for e in all_events if e["type"] == "SIGNAL_SKIPPED"]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Entries: {len(entries)}  Exits: {len(exits)}  Skipped: {len(skipped)}")
    wins = [e for e in exits if e["result"] == "WIN"]
    win_rate = len(wins) / len(exits) * 100 if exits else 0
    print(f"Win rate: {win_rate:.1f}% ({len(wins)}W / {len(exits)-len(wins)}L)")

    exit_reason_counts = {}
    for e in exits:
        exit_reason_counts[e["reason"]] = exit_reason_counts.get(e["reason"], 0) + 1
    print(f"Exit reasons: {exit_reason_counts}")

    if exits:
        losses = [e for e in exits if e["result"] == "LOSS"]
        if losses:
            worst_loss = min(losses, key=lambda e: e["net_pnl_inr"])
            print(f"Worst single-trade loss: Rs.{worst_loss['net_pnl_inr']:.2f} "
                  f"({worst_loss['direction']}, reason: {worst_loss['reason']}, "
                  f"entered {worst_loss['entry_time']}, closed {worst_loss['exit_time']})")
        durations_hours = [(pd.Timestamp(e["exit_time"]) - pd.Timestamp(e["entry_time"])).total_seconds() / 3600
                            for e in exits]
        print(f"Trade duration: median {pd.Series(durations_hours).median():.1f}h, "
              f"longest {max(durations_hours):.1f}h")

    print(f"\nFinal balance: Rs.{state['account_balance']:.2f}")
    print(f"Total withdrawn: Rs.{state['total_withdrawn']:.2f}")
    total_wealth = state["account_balance"] + state["total_withdrawn"]
    print(f"Total wealth: Rs.{total_wealth:.2f}")

    if state["open_positions"]:
        print(f"\nStill open at end of data: {len(state['open_positions'])} position(s)")
        for pos in state["open_positions"]:
            be = pos.get("breakeven_stop_price")
            print(f"  {pos['direction']} entered {pos['entry_time']} @ Rs.{pos['entry_price']:.2f}, "
                  f"EMA150 stop Rs.{pos['stop_price']:.2f}"
                  + (f", breakeven armed @ Rs.{be:.2f}" if be is not None else ", breakeven not armed"))

    trade_log_df = build_backtest_trade_log_df(all_events)

    # FIX: build_backtest_trade_log_df only processes ENTRY/EXIT events, so
    # a position that's STILL OPEN at the end of the backtest never gets
    # its target_2_revised (breakeven-armed) state reflected in the report
    # -- that state only exists in memory on state["open_positions"], since
    # append_trade_log() is a no-op during backtesting (avoids thousands of
    # real disk writes during simulation) and the events list has no event
    # type for "breakeven was armed mid-trade, position still open".
    # Confirmed directly: a real SHORT trade with breakeven genuinely armed
    # (per the console summary above) showed target_2_revised as blank in
    # the report before this fix. Patch it here using the live in-memory
    # state, which DOES correctly hold each open position's current
    # breakeven_stop_price.
    for pos in state["open_positions"]:
        be = pos.get("breakeven_stop_price")
        if be is not None:
            match = trade_log_df["entry_time"].astype(str) == str(pos["entry_time"])
            if match.any():
                trade_log_df.loc[match, "target_2_revised"] = round(be, 2)

    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)
    trade_log_df.to_csv(BACKTEST_OUTPUT_CSV, index=False)
    print(f"\nFull trade log (CSV): {BACKTEST_OUTPUT_CSV}")

    html = build_html_report(state, all_events[-50:] if len(all_events) > 50 else all_events, trade_log_df)
    html = _inject_exit_reason_column(html, trade_log_df)
    banner = f"""
    <div style="background:#1e3a5f; border:1px solid #2563eb; color:#93c5fd; padding:12px 16px;
                border-radius:8px; font-size:13px; margin-bottom:16px;">
        📊 ROLLING-WINDOW BACKTEST -- live entry/EMA150-stop logic + breakeven-hold exit rule +
        multiple simultaneous positions. Each simulated check re-slices a trailing
        {ROLLING_WINDOW_DAYS}-day window, matching what the real deployment would see. NOT a live
        report -- generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.
    </div>"""
    html = html.replace("<div class=\"warning\">", banner + "\n    <div class=\"warning\">")

    with open(BACKTEST_OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Full HTML report: {BACKTEST_OUTPUT_HTML}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if "--backtest" in args:
        run_backtest()
    elif "--loop-minutes" in args:
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