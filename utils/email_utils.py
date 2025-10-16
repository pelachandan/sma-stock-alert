import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

def format_summary(sma_list, high_list):
    """
    Builds a detailed summary for email content.
    """
    summary = ""

    if sma_list:
        summary += "📈 **SMA Crossovers (5–10% above crossover)**\n\n"
        for s in sma_list:
            trend_tag = (
                "🔥 Strong Momentum" if s["PctAbove"] >= 8 else "⚡ Steady Trend"
            )
            summary += (
                f"- {s['ticker']}: +{s['PctAbove']}% above crossover "
                f"(Crossed {s['CrossoverDate']}, ${s['CrossoverPrice']} → ${s['CurrentPrice']}) "
                f"{trend_tag}\n"
            )
        summary += "\n"

    if high_list:
        summary += "🚀 **New 52-Week Highs**\n\n"
        for ticker in high_list:
            summary += f"- {ticker}\n"

    if not summary:
        summary = "No new signals today."

    return summary


def send_email_alert(sma_list, high_list, subject_prefix="📊 Market Summary", custom_body=None):
    """
    Sends an email with either a custom body or formatted summary of SMA and high signals.
    """
    body = custom_body or format_summary(sma_list, high_list)

    sender = os.getenv("EMAIL_SENDER")
    receiver = os.getenv("EMAIL_RECEIVER")
    password = os.getenv("EMAIL_PASSWORD")

    subject = f"{subject_prefix} – {datetime.now().strftime('%Y-%m-%d')}"

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email sent: {subject}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
