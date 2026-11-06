"""Dependency-light CPU contract runner.

It runs the same functions as ``tests/test_cpu.py`` without importing pytest,
which is useful on a clean training node. No CUDA APIs are invoked.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

from .tests import test_cpu


def main():
    passed = 0
    for name, function in sorted(vars(test_cpu).items()):
        if not name.startswith("test_") or not callable(function):
            continue
        with tempfile.TemporaryDirectory() as directory:
            signature = inspect.signature(function)
            if signature.parameters:
                function(Path(directory))
            else:
                function()
        print(f"PASS {name}")
        passed += 1
    print(f"{passed} CPU tests passed")


if __name__ == "__main__":
    main()
