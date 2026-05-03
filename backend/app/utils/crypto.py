"""Fernet-based encryption for stored API keys. Key auto-generates on first run
and is persisted under data/.encryption_key (chmod 0600 best-effort)."""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet

from app.utils.config import settings


_KEY_FILE = settings.data_dir / ".encryption_key"


def _load_or_create_key() -> bytes:
    if settings.encryption_key:
        return settings.encryption_key.encode()
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes().strip()
    key = Fernet.generate_key()
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_bytes(key)
    try:
        os.chmod(_KEY_FILE, 0o600)
    except OSError:
        pass
    return key


_fernet = Fernet(_load_or_create_key())


def encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    return _fernet.decrypt(ciphertext.encode()).decode()


def mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****" + value[-2:]
    return "****" + value[-4:]
