import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from config import EMAIL_SENDER, EMAIL_RECEIVER, EMAIL_PASSWORD

def send_email_alert(symbols, subject_prefix="Daily Alert", custom_body=None):
    if custom_body:
        body = custom_body
    elif not symbols:
        body = "No signals today."
    else:
        body = "Tickers:\n" + "\n".join(symbols)

    subject = f"{subject_prefix} – {datetime.now().strftime('%Y-%m-%d')}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
