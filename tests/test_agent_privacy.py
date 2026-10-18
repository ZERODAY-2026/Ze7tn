import os
import unittest
from unittest.mock import patch

from src.gemma_agent import _ensure_disclaimer, _lock_localhost
from src.prompts import LEGAL_DISCLAIMER


class PrivacyTests(unittest.TestCase):
    def test_localhost_lock_rejects_remote_host(self):
        with patch.dict(os.environ, {}, clear=False):
            import src.gemma_agent as ga

            original = ga.OLLAMA_HOST
            try:
                ga.OLLAMA_HOST = "https://api.example.com:11434"
                with self.assertRaises(RuntimeError):
                    ga._lock_localhost()
            finally:
                ga.OLLAMA_HOST = original

    def test_disclaimer_appended_when_model_omits_it(self):
        text = _ensure_disclaimer("### What the document explicitly says\nHello")
        self.assertIn(LEGAL_DISCLAIMER, text)


if __name__ == "__main__":
    unittest.main()
