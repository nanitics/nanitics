"""Self-improver showcase validation — real-LLM smoke coverage for the critic phase.

``docker/full-stack/`` is not a Python package, but each runner inside it is.
Adds that directory to ``sys.path`` so ``self_improver.advisor`` imports here
the same way it does in the compose image.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FULL_STACK_DIR = Path(__file__).resolve().parent.parent.parent / "docker" / "full-stack"
if str(_FULL_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_STACK_DIR))
