from __future__ import annotations

import hashlib
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from egx_research.utils import ensure_dir, write_json


def sign_payload(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_payload(payload: bytes, signature: str, secret: str) -> bool:
    supplied = signature.removeprefix("sha256=")
    return hmac.compare_digest(sign_payload(payload, secret), supplied)


def accept_webhook(payload: bytes, signature: str, secret: str, store_dir: str | Path) -> dict[str, Any]:
    if not verify_payload(payload, signature, secret):
        raise ValueError("Invalid webhook signature")
    body = json.loads(payload)
    event_id = str(body.get("id") or hashlib.sha256(payload).hexdigest())
    target = ensure_dir(store_dir) / f"{event_id}.json"
    duplicate = target.exists()
    if not duplicate:
        write_json(target, body)
    return {"accepted": True, "duplicate": duplicate, "event_id": event_id, "path": str(target)}


def serve_webhooks(host: str, port: int, secret: str, store_dir: str | Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                self.send_error(413)
                return
            payload = self.rfile.read(length)
            signature = self.headers.get("X-EGX-Signature", self.headers.get("X-Signature", ""))
            try:
                result = accept_webhook(payload, signature, secret, store_dir)
                encoded = json.dumps(result).encode("utf-8")
                self.send_response(200)
            except (ValueError, json.JSONDecodeError) as exc:
                encoded = json.dumps({"accepted": False, "error": str(exc)}).encode("utf-8")
                self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
