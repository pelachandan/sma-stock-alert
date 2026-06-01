import tempfile
from pathlib import Path

import pandas as pd

from src.notifications import email as email_module


class _FakeSMTP:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.login_args = None
        self.sendmail_args = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login(self, sender, password):
        self.login_args = (sender, password)

    def sendmail(self, sender, recipients, message):
        self.sendmail_args = (sender, recipients, message)


def test_send_email_alert_uses_email_config_bcc_recipients(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="email-config-test-") as tmp_dir:
        config_path = Path(tmp_dir) / "emailConfig.json"
        config_path.write_text(
            '{"recipients": ["alice@example.com", "bob@example.com"]}',
            encoding="utf-8",
        )

        fake_smtp = _FakeSMTP("smtp.gmail.com", 465)
        monkeypatch.setattr(email_module, "LOCAL_EMAIL_CONFIG_PATHS", (config_path,))
        monkeypatch.setattr(email_module, "GCS_EMAIL_CONFIG_PATHS", ())
        monkeypatch.setattr(email_module.smtplib, "SMTP_SSL", lambda host, port: fake_smtp)
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.delenv("EMAIL_RECEIVER", raising=False)
        monkeypatch.delenv("EMAIL_RECIPIENTS", raising=False)

        email_module.send_email_alert(
            trade_df=pd.DataFrame(),
            all_signals=[],
            subject_prefix="Test Email",
        )

        assert fake_smtp.login_args == ("sender@example.com", "secret")
        assert fake_smtp.sendmail_args is not None
        sender, recipients, message = fake_smtp.sendmail_args
        assert sender == "sender@example.com"
        assert recipients == ["alice@example.com", "bob@example.com"]
        assert "To: sender@example.com" in message
        assert "alice@example.com" not in message
        assert "bob@example.com" not in message


def test_send_email_alert_falls_back_to_env_recipients(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="email-config-test-") as tmp_dir:
        config_path = Path(tmp_dir) / "emailConfig.json"
        config_path.write_text('{"recipients": []}', encoding="utf-8")

        fake_smtp = _FakeSMTP("smtp.gmail.com", 465)
        monkeypatch.setattr(email_module, "LOCAL_EMAIL_CONFIG_PATHS", (config_path,))
        monkeypatch.setattr(email_module, "GCS_EMAIL_CONFIG_PATHS", ())
        monkeypatch.setattr(email_module.smtplib, "SMTP_SSL", lambda host, port: fake_smtp)
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setenv("EMAIL_RECEIVER", "carol@example.com, dave@example.com")

        email_module.send_email_alert(
            trade_df=pd.DataFrame(),
            all_signals=[],
            subject_prefix="Fallback Email",
        )

        assert fake_smtp.sendmail_args is not None
        assert fake_smtp.sendmail_args[1] == ["carol@example.com", "dave@example.com"]


def test_load_email_config_uses_gcs_when_local_template_is_empty(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="email-config-test-") as tmp_dir:
        local_config_path = Path(tmp_dir) / "emailConfig.json"
        local_config_path.write_text('{"recipients": []}', encoding="utf-8")

        def fake_download_file(gcs_path, local_path):
            Path(local_path).write_text(
                '{"recipients": ["gcs1@example.com", "gcs2@example.com"]}',
                encoding="utf-8",
            )
            return True

        monkeypatch.setattr(email_module, "LOCAL_EMAIL_CONFIG_PATHS", (local_config_path,))
        monkeypatch.setattr(email_module, "GCS_EMAIL_CONFIG_PATHS", ("config/emailConfig.json",))
        monkeypatch.setattr(email_module, "download_file", fake_download_file)

        config = email_module._load_email_config()

        assert config["recipients"] == ["gcs1@example.com", "gcs2@example.com"]
