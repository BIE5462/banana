from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.fernet import Fernet
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

CONFIG_SEED = b"NanoBanana::License::Config::v1"
CONFIG_FERNET_KEY = base64.urlsafe_b64encode(hashlib.sha256(CONFIG_SEED).digest())


def get_fernet() -> Fernet:
    return Fernet(CONFIG_FERNET_KEY)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encrypt_text(plain_text: str) -> str:
    encrypted = get_fernet().encrypt(plain_text.encode("utf-8"))
    return encrypted.decode("utf-8")


def decrypt_text(cipher_text: str) -> str:
    decrypted = get_fernet().decrypt(cipher_text.encode("utf-8"))
    return decrypted.decode("utf-8")


def try_decrypt_text(cipher_text: str) -> tuple[bool, str]:
    try:
        return True, decrypt_text(cipher_text)
    except Exception:
        return False, ""


def encrypt_config_payload(payload: dict[str, Any]) -> bytes:
    raw_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return get_fernet().encrypt(raw_text.encode("utf-8"))


def decrypt_config_payload(cipher_bytes: bytes) -> dict[str, Any]:
    plain_text = get_fernet().decrypt(cipher_bytes).decode("utf-8")
    data = json.loads(plain_text)
    if not isinstance(data, dict):
        raise ValueError("授权配置内容必须为对象")
    return data


def encrypt_aes_ecb_pkcs5(input_string: str, key: str) -> str:
    key_bytes = key.encode("utf-8")
    if len(key_bytes) not in {16, 24, 32}:
        raise ValueError("encrypt_key 长度必须为 16、24 或 32 字节")
    cipher = AES.new(key_bytes, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(input_string.encode("utf-8"), AES.block_size))
    return encrypted.hex()


def decrypt_aes_ecb_pkcs5(encrypted_hex: str, key: str) -> str:
    key_bytes = key.encode("utf-8")
    if len(key_bytes) not in {16, 24, 32}:
        raise ValueError("encrypt_key 长度必须为 16、24 或 32 字节")
    cipher = AES.new(key_bytes, AES.MODE_ECB)
    decrypted = cipher.decrypt(bytes.fromhex(encrypted_hex))
    return unpad(decrypted, AES.block_size).decode("utf-8")
