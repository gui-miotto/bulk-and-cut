import unittest
import subprocess
import os


class TestPEP8(unittest.TestCase):

    def test_PEP8(self):
        here = os.path.abspath(os.path.dirname(__file__))
        flake_cmd = [
            "flake8",
            "--max-line-length=100",
            "--show-source",
            os.path.join(here, "..", "bulkandcut"),
            # os.path.join(here, "..", "examples"),  # TODO: uncoment
            os.path.join(here, "..", "test"),
        ]

        process = subprocess.Popen(
            flake_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            )
        stdout, stderr = process.communicate()
        if process.poll() != 0:
            print(stdout)
            raise RuntimeError(stderr)


if __name__ == '__main__':
    unittest.main()
