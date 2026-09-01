import json
import os
import smtplib
import tempfile
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from src.config.settings import POSITION_INITIAL_EQUITY, POSITION_RISK_PER_TRADE_PCT
from src.storage.gcs import download_file


LOCAL_EMAIL_CONFIG_PATHS = (
    Path("config") / "emailConfig.json",
    Path("config") / "emailConfig",
)
GCS_EMAIL_CONFIG_PATHS = (
    "config/emailConfig.json",
    "config/emailConfig",
)


def _normalize_email_list(raw_value):
    if raw_value is None:
        return []

    if isinstance(raw_value, str):
        values = raw_value.replace(";", ",").split(",")
    elif isinstance(raw_value, list):
        values = []
        for item in raw_value:
            if isinstance(item, str):
                values.extend(item.replace(";", ",").split(","))
    else:
        return []

    seen = set()
    normalized = []
    for value in values:
        email = value.strip()
        if not email or email in seen:
            continue
        seen.add(email)
        normalized.append(email)
    return normalized


def _parse_email_config(data):
    if isinstance(data, list):
        return {"sender": None, "password": None, "recipients": _normalize_email_list(data), "strategy_recipients": {}}

    if not isinstance(data, dict):
        raise ValueError("Email config must be a JSON object or array of recipient addresses.")

    recipients = []
    for key in ("bcc_recipients", "recipients", "email_ids", "emails"):
        if key in data:
            recipients = _normalize_email_list(data.get(key))
            break

    sender = str(data.get("sender") or data.get("from") or "").strip() or None
    password = str(data.get("password") or data.get("app_password") or "").strip() or None
    strategy_recipients = {
        str(strategy): _normalize_email_list(addresses)
        for strategy, addresses in data.get("strategy_recipients", {}).items()
    }
    return {"sender": sender, "password": password, "recipients": recipients, "strategy_recipients": strategy_recipients}


def _read_email_config_file(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        return _parse_email_config(json.load(handle))


def _has_email_config_values(config):
    return bool(config["sender"] or config["password"] or config["recipients"] or config["strategy_recipients"])


def _load_email_config():
    for local_path in LOCAL_EMAIL_CONFIG_PATHS:
        if not local_path.exists():
            continue
        try:
            config = _read_email_config_file(local_path)
            print(f"⚙️  [email] Loaded email config from local {local_path}")
            if _has_email_config_values(config):
                return config
            print(f"ℹ️  [email] Local email config {local_path} is empty; checking GCS fallback")
        except Exception as exc:
            print(f"⚠️  [email] Could not load local email config {local_path}: {exc}")

    with tempfile.TemporaryDirectory(prefix="email-config-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for gcs_path in GCS_EMAIL_CONFIG_PATHS:
            try:
                local_path = tmp_root / Path(gcs_path).name
                if not download_file(gcs_path, local_path):
                    continue
                config = _read_email_config_file(local_path)
                print(f"⚙️  [email] Loaded email config from GCS {gcs_path}")
                if _has_email_config_values(config):
                    return config
                print(f"ℹ️  [email] GCS email config {gcs_path} is empty")
            except Exception as exc:
                print(f"⚠️  [email] Could not load GCS email config {gcs_path}: {exc}")

    return {"sender": None, "password": None, "recipients": [], "strategy_recipients": {}}


# ============================================================
# Helper: Create HTML table with score-based row coloring
# ============================================================
def df_to_html_table(df, score_column="Score", title="", max_rows=5):
    if df is None or df.empty:
        return f"<p>No {title} today.</p>"

    # --- Sort by score descending if score exists ---
    if score_column and score_column in df.columns:
        df = df.sort_values(by=score_column, ascending=False)

    # --- Limit rows ---
    if max_rows is not None:
        df = df.head(max_rows)

    html = f"<h2>{title}</h2>"
    html += "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;'>"

    # Header
    html += "<tr>"
    for col in df.columns:
        html += (
            "<th style='background-color:#f2f2f2;"
            "font-weight:bold;text-align:center;'>"
            f"{col}</th>"
        )
    html += "</tr>"

    # Rows
    for _, row in df.iterrows():
        score = row.get(score_column, 0) if score_column else 0

        if score_column:
            if score >= 8.5:
                color = "#c6efce"   # green
            elif score >= 6.5:
                color = "#ffeb9c"   # yellow
            else:
                color = "#f4c7c3"   # red
        else:
            color = "#ffffff"      # neutral

        html += f"<tr style='background-color:{color};'>"
        for col in df.columns:
            # Show numbers rounded if float
            val = row[col]
            if isinstance(val, float):
                val = round(val, 2)
            html += f"<td style='text-align:center;'>{val}</td>"
        html += "</tr>"

    html += "</table><br>"
    return html


# ============================================================
# Normalize lists for table-friendly DataFrames
# ============================================================
def normalize_highs_for_table(high_list):
    if not high_list:
        return pd.DataFrame()
    df = pd.DataFrame(high_list)
    preferred_columns = [
        "Ticker", "Company", "Close", "High52", "PctFrom52High",
        "EMA20", "EMA50", "EMA200", "VolumeRatio", "RSI14", "Score", "NormalizedScore",
    ]
    return df[[c for c in preferred_columns if c in df.columns]]


def normalize_watchlist_for_table(watch_list):
    if not watch_list:
        return pd.DataFrame()
    df = pd.DataFrame(watch_list)
    preferred_columns = [
        "Ticker", "Company", "Close", "High52", "PctFrom52High",
        "EMA20", "EMA50", "EMA200", "RSI14",
    ]
    return df[[c for c in preferred_columns if c in df.columns]]


def normalize_generic_for_table(generic_list):
    """For consolidation_list or rs_list where Score/NormalizedScore may exist"""
    if not generic_list:
        return pd.DataFrame()
    return pd.DataFrame(generic_list)  # keep all columns


# ============================================================
# Main Email Sender
# ============================================================
def send_email_alert(
    trade_df,
    all_signals=None,
    high_buy_list=None,
    high_watch_list=None,
    ema_list=None,
    consolidation_list=None,
    rs_list=None,
    subject_prefix="📊 Market Summary",
    html_body=None,
    position_tracker=None,
    action_signals=None,  # 🆕 NEW parameter for exit/action alerts
):
    """
    Sends simplified HTML email with:
    - Open positions (currently held)
    - Actionable trades only (top trades that passed all filters)
    - Watchlist of 10 stocks with strategy names
    """
    if html_body:
        body_html = html_body
    else:
        # Detect if this is position trading
        is_position_trading = not trade_df.empty and "Priority" in trade_df.columns

        if is_position_trading:
            body_html = "<h1>📊 Position Trading Scanner - Long-Term Setups</h1>"
            body_html += f"<p>{datetime.now().strftime('%Y-%m-%d')}</p>"
        else:
            body_html = "<h1>📊 Daily Market Scan - Actionable Trades</h1>"
            body_html += f"<p>{datetime.now().strftime('%Y-%m-%d')}</p>"

        # 🚨 ACTION SIGNALS (HIGHEST PRIORITY - SHOW FIRST)
        if action_signals:
            exits = action_signals.get('exits', [])
            partials = action_signals.get('partials', [])
            pyramids = action_signals.get('pyramids', [])
            warnings = action_signals.get('warnings', [])

            total_actions = len(exits) + len(partials) + len(pyramids)

            if total_actions > 0:
                body_html += "<hr style='border: 3px solid red;'>"
                body_html += f"<h1 style='color: red;'>🚨 ACTION REQUIRED: {total_actions} SIGNAL(S)</h1>"

                # EXITS (Most Urgent)
                if exits:
                    body_html += "<h2 style='color: red;'>🚨 IMMEDIATE EXITS ({}) - ACTION REQUIRED</h2>".format(len(exits))
                    body_html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse; width:100%;'>"
                    body_html += "<tr style='background-color:#ffcccc;'>"
                    body_html += "<th>Ticker</th><th>Exit Type</th><th>Reason</th><th>Action</th><th>Current R</th><th>Days</th><th>Entry $</th><th>Current $</th></tr>"

                    for exit_sig in exits:
                        urgency_color = "#ff0000" if exit_sig['urgency'] == 'IMMEDIATE' else "#ff9999"
                        body_html += f"<tr style='background-color:{urgency_color};'>"
                        body_html += f"<td><strong>{exit_sig['ticker']}</strong></td>"
                        body_html += f"<td>{exit_sig['type']}</td>"
                        body_html += f"<td>{exit_sig['reason']}</td>"
                        body_html += f"<td><strong>{exit_sig['action']}</strong></td>"
                        body_html += f"<td>{exit_sig['current_r']:+.2f}R</td>"
                        body_html += f"<td>{exit_sig['days_held']}d</td>"
                        body_html += f"<td>${exit_sig['entry_price']:.2f}</td>"
                        body_html += f"<td>${exit_sig['current_price']:.2f}</td>"
                        body_html += "</tr>"

                    body_html += "</table><br>"

                # PARTIAL PROFITS
                if partials:
                    body_html += "<h2 style='color: green;'>💰 PARTIAL PROFIT TARGETS ({}) - TAKE PROFITS</h2>".format(len(partials))
                    body_html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse; width:100%;'>"
                    body_html += "<tr style='background-color:#ccffcc;'>"
                    body_html += "<th>Ticker</th><th>Reason</th><th>Action</th><th>Current R</th><th>Days</th><th>Entry $</th><th>Current $</th></tr>"

                    for partial in partials:
                        body_html += "<tr style='background-color:#e6ffe6;'>"
                        body_html += f"<td><strong>{partial['ticker']}</strong></td>"
                        body_html += f"<td>{partial['reason']}</td>"
                        body_html += f"<td><strong>{partial['action']}</strong></td>"
                        body_html += f"<td>+{partial['current_r']:.2f}R</td>"
                        body_html += f"<td>{partial['days_held']}d</td>"
                        body_html += f"<td>${partial['entry_price']:.2f}</td>"
                        body_html += f"<td>${partial['current_price']:.2f}</td>"
                        body_html += "</tr>"

                    body_html += "</table><br>"

                # PYRAMID OPPORTUNITIES
                if pyramids:
                    body_html += "<h2 style='color: blue;'>📈 PYRAMID OPPORTUNITIES ({}) - ADD TO WINNERS</h2>".format(len(pyramids))
                    body_html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse; width:100%;'>"
                    body_html += "<tr style='background-color:#cce5ff;'>"
                    body_html += "<th>Ticker</th><th>Reason</th><th>Action</th><th>Current R</th><th>Days</th><th>Entry $</th><th>Current $</th></tr>"

                    for pyramid in pyramids:
                        body_html += "<tr style='background-color:#e6f2ff;'>"
                        body_html += f"<td><strong>{pyramid['ticker']}</strong></td>"
                        body_html += f"<td>{pyramid['reason']}</td>"
                        body_html += f"<td><strong>{pyramid['action']}</strong></td>"
                        body_html += f"<td>+{pyramid['current_r']:.2f}R</td>"
                        body_html += f"<td>{pyramid['days_held']}d</td>"
                        body_html += f"<td>${pyramid['entry_price']:.2f}</td>"
                        body_html += f"<td>${pyramid['current_price']:.2f}</td>"
                        body_html += "</tr>"

                    body_html += "</table><br>"

                body_html += "<hr style='border: 3px solid red;'>"

        # 🆕 Show Open Positions (if any)
        if position_tracker:
            open_positions = position_tracker.get_all_positions()
            if open_positions:
                pos_data = []
                for ticker, pos in open_positions.items():
                    pos_data.append({
                        "Ticker": ticker,
                        "Entry $": f"${pos.get('entry_price', 0):.2f}",
                        "Entry Date": str(pos.get('entry_date', 'N/A'))[:10],
                        "Strategy": pos.get('strategy', 'Unknown'),
                        "Stop $": f"${pos.get('stop_loss', 0):.2f}" if pos.get('stop_loss', 0) > 0 else "N/A",
                        "Target $": f"${pos.get('target', 0):.2f}" if pos.get('target') and pos.get('target', 0) > 0 else "TRAIL MA100",
                    })

                pos_df = pd.DataFrame(pos_data)
                body_html += "<hr>"
                body_html += df_to_html_table(
                    pos_df,
                    score_column=None,
                    title=f"📊 CURRENT OPEN POSITIONS ({len(pos_df)} stocks)",
                    max_rows=None
                )
                body_html += "<hr>"

        # Actionable Trades Only
        if not trade_df.empty:
            # Detect if this is position trading or short-term trading
            is_position_trading = "Priority" in trade_df.columns and "MaxDays" in trade_df.columns

            if is_position_trading:
                # Calculate position sizing for each trade
                # User can update POSITION_INITIAL_EQUITY in config/trading_config.py
                equity = POSITION_INITIAL_EQUITY  # Default $100k
                risk_pct = POSITION_RISK_PER_TRADE_PCT / 100  # 2% default

                body_html += f"<h2>🎯 NEW POSITION TRADES ({len(trade_df)} signals) - ACTION ITEMS</h2>"
                body_html += f"<p><strong>Account Equity:</strong> ${equity:,.0f} | <strong>Risk per Trade:</strong> {risk_pct*100}% = ${equity*risk_pct:,.0f}</p>"
                body_html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse; width:100%;'>"
                body_html += "<tr style='background-color:#e6f2ff;'>"
                body_html += "<th>Ticker</th><th>Strategy</th><th>Action</th><th>Shares</th><th>Position $</th>"
                body_html += "<th>Entry $</th><th>Stop $</th><th>Target $</th><th>Risk/Share</th><th>Max Days</th></tr>"

                for idx, row in trade_df.iterrows():
                    ticker = row['Ticker']
                    strategy = row['Strategy']
                    if strategy == "Streak_Position":
                        probability = row.get("ProbabilityNextGreen", 0)
                        reason = row.get("PredictionReason", "Rolling ranker selection")
                        body_html += "<tr style='background-color:#c6efce;'>"
                        body_html += f"<td><strong>{ticker}</strong></td>"
                        body_html += f"<td>{strategy}</td>"
                        body_html += (
                            "<td colspan='8'><strong>OPTION GUIDANCE:</strong> Buy one 7-day ITM call "
                            "at the next session open; exit at that session close. "
                            f"Estimated next-day green probability: {probability:.1%}.<br>{reason}</td>"
                        )
                        body_html += "</tr>"
                        continue
                    entry = row['Entry']
                    stop = row['StopLoss']
                    target = row['Target']
                    max_days = row.get('MaxDays', 150)

                    # Calculate position sizing
                    risk_amount = equity * risk_pct  # $2,000 default
                    if entry > 0 and stop > 0:
                        risk_per_share = row.get("RiskPerShare")
                        if pd.isna(risk_per_share) or risk_per_share is None or risk_per_share <= 0:
                            risk_per_share = max(abs(entry - stop), entry * 0.01)
                        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                        shares = min(shares, int(equity * 0.25 / entry))  # 25% cap
                    else:
                        shares = 0
                    position_size = shares * entry

                    # Color code by score (0-100 scale for position trading)
                    score = row.get('Score', 0)
                    if score >= 80:
                        row_color = "#c6efce"  # green  — strong RS
                    elif score >= 50:
                        row_color = "#ffeb9c"  # yellow — moderate RS
                    else:
                        row_color = "#f4c7c3"  # red    — weak RS

                    body_html += f"<tr style='background-color:{row_color};'>"
                    body_html += f"<td><strong>{ticker}</strong></td>"
                    body_html += f"<td>{strategy}</td>"
                    body_html += f"<td><strong>BUY {shares} shares at ${entry:.2f}</strong></td>"
                    body_html += f"<td><strong>{shares}</strong></td>"
                    body_html += f"<td>${position_size:,.0f}</td>"
                    body_html += f"<td>${entry:.2f}</td>"
                    body_html += f"<td>${stop:.2f}</td>"
                    if target and target > 0:
                        body_html += f"<td>${target:.2f}</td>"
                    else:
                        body_html += "<td><em>TRAIL MA100</em></td>"
                    body_html += f"<td>${risk_per_share:.2f}</td>"
                    body_html += f"<td>{max_days}d</td>"
                    body_html += "</tr>"

                body_html += "</table><br>"

            else:
                # Short-term trading columns (old system)
                action_cols = ["Ticker", "Strategy", "Entry", "StopLoss", "Target",
                               "ATR", "SuggestedShares", "FinalScore", "Expectancy"]
                score_col = "FinalScore"
                title_prefix = "🔥 ACTIONABLE TRADES"

                action_df = trade_df[[c for c in action_cols if c in trade_df.columns]]

                body_html += df_to_html_table(
                    action_df,
                    score_column=score_col,
                    title=f"{title_prefix} ({len(action_df)} stocks)",
                    max_rows=None  # Show all actionable trades
                )
        else:
            body_html += "<h2>🔥 ACTIONABLE TRADES</h2><p>No actionable trades today.</p>"

        # Watchlist - Top 10 stocks that didn't make the cut
        watchlist_items = []

        # Combine all signals that aren't in actionable trades
        if all_signals:
            actionable_tickers = set(trade_df["Ticker"].tolist()) if not trade_df.empty else set()

            for signal in all_signals:
                ticker = signal.get("Ticker")
                if ticker and ticker not in actionable_tickers:
                    watchlist_items.append({
                        "Ticker": ticker,
                        "Strategy": signal.get("Strategy", "Unknown"),
                        "Score": signal.get("Score", 0)
                    })

        # Sort by score and take top 10
        if watchlist_items:
            watch_df = pd.DataFrame(watchlist_items)
            watch_df = watch_df.sort_values(by="Score", ascending=False).head(10)
            body_html += df_to_html_table(
                watch_df,
                score_column="Score",
                title="👀 WATCHLIST (Top 10 Non-Actionable Signals)",
                max_rows=None
            )
        else:
            body_html += "<h2>👀 WATCHLIST</h2><p>No watchlist stocks today.</p>"

    email_config = _load_email_config()

    # Sender credentials continue to support env fallback so secrets stay out of git.
    sender = (
        email_config["sender"]
        or os.getenv("EMAIL_SENDER")
        or os.getenv("EMAIL_FROM")
    )
    password = (
        email_config["password"]
        or os.getenv("EMAIL_PASSWORD")
        or os.getenv("SMTP_PASSWORD")
    )
    strategy_names = set(trade_df["Strategy"]) if not trade_df.empty and "Strategy" in trade_df else set()
    strategy_receivers = (
        email_config["strategy_recipients"].get(next(iter(strategy_names)), [])
        if len(strategy_names) == 1
        else []
    )
    receivers = (
        strategy_receivers
        or
        email_config["recipients"]
        or _normalize_email_list(os.getenv("EMAIL_RECEIVER", ""))
        or _normalize_email_list(os.getenv("EMAIL_RECIPIENTS", ""))
    )

    if not sender:
        print("❌ Failed to send email: missing sender address (EMAIL_SENDER or emailConfig sender)")
        return
    if not password:
        print("❌ Failed to send email: missing email password (EMAIL_PASSWORD or emailConfig password)")
        return
    if not receivers:
        print("❌ Failed to send email: no recipient addresses configured in emailConfig or env")
        return

    # Update subject if there are urgent actions
    subject = f"{subject_prefix} – {datetime.now().strftime('%Y-%m-%d')}"
    if action_signals:
        total_actions = len(action_signals.get('exits', [])) + len(action_signals.get('partials', []))
        if total_actions > 0:
            subject = f"🚨 ACTION REQUIRED ({total_actions}) – {subject}"

    # Build MIME email
    msg = MIMEText(body_html, "html")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = sender

    # Recipients are delivered via SMTP envelope only so they stay hidden from one another.
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())
        print(f"✅ Email sent: {subject}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
