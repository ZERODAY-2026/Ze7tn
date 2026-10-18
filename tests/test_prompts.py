import unittest

from src.prompts import LEGAL_DISCLAIMER, SYSTEM_PROMPT, build_user_prompt


class PromptTests(unittest.TestCase):
    def test_system_prompt_requires_grounding_and_split(self):
        text = SYSTEM_PROMPT.lower()
        self.assertIn("not a lawyer", text)
        self.assertIn("explicitly says", text)
        self.assertIn("interpretation", text)
        self.assertTrue("cite" in text or "citation" in text)
        self.assertIn("refuse", text)
        self.assertIn(LEGAL_DISCLAIMER.lower(), text)

    def test_user_prompt_embeds_passages_and_question(self):
        prompt = build_user_prompt(
            "What is the duration?",
            [
                {
                    "id": "C1",
                    "page": 1,
                    "start_char": 0,
                    "text": "Obligations continue for two years.",
                }
            ],
            document_name="sample_nda.txt",
        )
        self.assertIn("What is the duration?", prompt)
        self.assertIn("C1", prompt)
        self.assertIn("two years", prompt)
        self.assertIn(LEGAL_DISCLAIMER, prompt)

    def test_disclaimer_wording_is_stable(self):
        self.assertEqual(
            LEGAL_DISCLAIMER,
            "This is not legal advice – consult a qualified lawyer.",
        )


if __name__ == "__main__":
    unittest.main()
