from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import requests

from egx_research.utils import write_json


def _send_channel(channel: str, message: str, notification: dict[str, Any], webhook_url: str | None = None) -> int:
    if channel in {"webhook", "slack", "discord"}:
        url = webhook_url or os.getenv("EGX_TV_WEBHOOK_URL")
        if not url:
            raise ValueError("Sending requires --webhook-url or EGX_TV_WEBHOOK_URL")
        payload = notification if channel == "webhook" else ({"text": message} if channel == "slack" else {"content": message})
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return response.status_code
    if channel == "telegram":
        token = os.getenv("EGX_TV_TELEGRAM_TOKEN")
        chat_id = os.getenv("EGX_TV_TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise ValueError("Telegram requires EGX_TV_TELEGRAM_TOKEN and EGX_TV_TELEGRAM_CHAT_ID")
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message}, timeout=20)
        response.raise_for_status()
        return response.status_code
    if channel == "email":
        host = os.getenv("EGX_TV_SMTP_HOST")
        sender = os.getenv("EGX_TV_EMAIL_FROM")
        recipient = os.getenv("EGX_TV_EMAIL_TO")
        if not host or not sender or not recipient:
            raise ValueError("Email requires EGX_TV_SMTP_HOST, EGX_TV_EMAIL_FROM, and EGX_TV_EMAIL_TO")
        mail = EmailMessage()
        mail["Subject"] = f"EGX TradingView signal: {notification['run_id']}"
        mail["From"] = sender
        mail["To"] = recipient
        mail.set_content(message)
        port = int(os.getenv("EGX_TV_SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=20) as client:
            client.starttls()
            username = os.getenv("EGX_TV_SMTP_USERNAME")
            password = os.getenv("EGX_TV_SMTP_PASSWORD")
            if username and password:
                client.login(username, password)
            client.send_message(mail)
        return 250
    raise ValueError(f"Unknown notification channel: {channel}")


def notify_run(
    run_id: str,
    runs_dir: str = "runs",
    send: bool = False,
    webhook_url: str | None = None,
    channel: str = "webhook",
) -> Path:
    run_dir = Path(runs_dir) / run_id
    source = run_dir / "current_signal.json"
    if not source.exists():
        source = run_dir / "scan.json"
    if not source.exists():
        raise FileNotFoundError(f"No scan or current signal found under {run_dir}")
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    message = f"{payload.get('strategy_id', 'strategy')} {payload.get('logical_symbol', '')}: exposure={payload.get('target_exposure', 'n/a')} as_of={payload.get('as_of', 'n/a')} action={payload.get('action', 'n/a')}"
    notification = {"run_id": run_id, "message": message, "payload": payload, "channel": channel, "sent": False}
    if send:
        status_code = _send_channel(channel, message, notification, webhook_url)
        notification["sent"] = True
        notification["status_code"] = status_code
    target = run_dir / "notification.json"
    write_json(target, notification)
    return target
