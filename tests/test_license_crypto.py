from __future__ import annotations

import unittest

from license_crypto import (
    decrypt_aes_ecb_pkcs5,
    decrypt_config_payload,
    encrypt_aes_ecb_pkcs5,
    encrypt_config_payload,
    encrypt_text,
    try_decrypt_text,
)


class LicenseCryptoTests(unittest.TestCase):
    def test_encrypt_text_round_trip(self) -> None:
        encrypted = encrypt_text("CARD-001")
        success, plain_text = try_decrypt_text(encrypted)

        self.assertTrue(success)
        self.assertEqual(plain_text, "CARD-001")

    def test_encrypt_config_payload_round_trip(self) -> None:
        payload = {"license_options": {"enabled": True, "app_id": "demo"}}

        encrypted = encrypt_config_payload(payload)
        decrypted = decrypt_config_payload(encrypted)

        self.assertEqual(decrypted, payload)

    def test_encrypt_aes_round_trip(self) -> None:
        encrypted = encrypt_aes_ecb_pkcs5(
            "card=TEST&mac=DEVICE", "1234567890abcdef"
        )
        decrypted = decrypt_aes_ecb_pkcs5(encrypted, "1234567890abcdef")

        self.assertEqual(decrypted, "card=TEST&mac=DEVICE")

    def test_encrypt_aes_rejects_invalid_key_length(self) -> None:
        with self.assertRaises(ValueError):
            encrypt_aes_ecb_pkcs5("demo", "short")


if __name__ == "__main__":
    unittest.main()
