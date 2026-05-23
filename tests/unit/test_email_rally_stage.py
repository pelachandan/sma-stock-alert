from email import message_from_string

import pandas as pd

from src.notifications.email import format_rally_stage, send_email_alert


class _FakeSMTP:
    last_message = None

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login(self, sender, password):
        self.sender = sender
        self.password = password

    def sendmail(self, sender, receivers, message):
        _FakeSMTP.last_message = message


class _TrackerStub:
    def get_all_positions(self):
        return {
            "MSFT": {
                "entry_price": 400.0,
                "entry_date": "2026-05-20",
                "strategy": "RallyPattern_Position",
                "leadership_stage": "confirmed",
                "setup_type": "power_breakout",
                "stop_loss": 380.0,
                "target": 460.0,
            }
        }


def test_format_rally_stage_uses_setup_type_fallback():
    assert format_rally_stage("RallyPattern_Position", None, "emerging_leader_breakout") == "Emerging"
    assert format_rally_stage("RallyPattern_Position", None, "power_breakout") == "Established"
    assert format_rally_stage("High52_Position", "confirmed", "power_breakout") == ""


def test_send_email_alert_includes_rally_stage_for_new_trades_and_open_positions(monkeypatch):
    trade_df = pd.DataFrame(
        [
            {
                "Ticker": "NVDA",
                "Strategy": "RallyPattern_Position",
                "Entry": 100.0,
                "StopLoss": 95.0,
                "Target": 115.0,
                "Score": 88.0,
                "Priority": 6,
                "MaxDays": 120,
                "LeadershipStage": "emerging",
                "SetupType": "emerging_leader_breakout",
            }
        ]
    )

    monkeypatch.setenv("EMAIL_SENDER", "bot@example.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "password")
    monkeypatch.setenv("EMAIL_RECEIVER", "user@example.com")
    monkeypatch.setattr("smtplib.SMTP_SSL", _FakeSMTP)
    _FakeSMTP.last_message = None

    send_email_alert(
        trade_df=trade_df,
        subject_prefix="Test",
        position_tracker=_TrackerStub(),
    )

    message = message_from_string(_FakeSMTP.last_message)
    html = message.get_payload(decode=True).decode(message.get_content_charset() or "utf-8")

    assert "Rally Stage" in html
    assert "Emerging" in html
    assert "Established" in html
