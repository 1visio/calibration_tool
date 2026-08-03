import tempfile
import unittest
from pathlib import Path

from calibration_tool.io_utils import sha256_file


class IoUtilsTests(unittest.TestCase):
    def test_normalized_hash_ignores_windows_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"a\nb\n")
            crlf.write_bytes(b"a\r\nb\r\n")
            self.assertEqual(sha256_file(lf), sha256_file(crlf))
            self.assertNotEqual(
                sha256_file(lf, normalize_newlines=False),
                sha256_file(crlf, normalize_newlines=False),
            )


if __name__ == "__main__":
    unittest.main()
