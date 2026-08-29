from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def test_non_string_job_name_is_a_clean_input_error(self) -> None:
        payload = {
            "ram_bytes": 4096,
            "jobs": [{
                "name": 123,
                "weight_bytes": 1,
                "context": 1,
                "layers": 1,
                "kv_heads": 1,
                "head_dim": 1,
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            spec = Path(temporary) / "bad-name.json"
            spec.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "bytebanklm", str(spec)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("job name must be a non-empty string", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
