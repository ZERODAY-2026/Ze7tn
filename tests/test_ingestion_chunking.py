import unittest
from pathlib import Path

from src.chunking import chunk_document
from src.ingestion import extract_file
from src.retrieval import MemoryIndex

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "sample_nda.txt"


class IngestionChunkingTests(unittest.TestCase):
    def test_extract_sample_nda(self):
        doc = extract_file(EXAMPLES)
        self.assertIn("Confidential Information", doc.text)
        self.assertTrue(doc.pages)
        self.assertEqual(doc.warnings, [])

    def test_chunks_have_stable_ids_and_pages(self):
        doc = extract_file(EXAMPLES)
        chunks = chunk_document(doc, chunk_size=400, overlap=40)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].id, "C1")
        self.assertTrue(all(c.page >= 1 for c in chunks))
        self.assertTrue(all(c.text.strip() for c in chunks))

    def test_lexical_retrieval_finds_duration_clause(self):
        doc = extract_file(EXAMPLES)
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        hits = MemoryIndex(chunks).query(
            "confidentiality period duration two years", k=3
        )
        joined = " ".join(h.chunk.text.lower() for h in hits)
        self.assertTrue("two (2) years" in joined or "two years" in joined)


if __name__ == "__main__":
    unittest.main()
