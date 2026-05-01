"""Tests for the ``self_improver`` showcase runner and its bundled advisor subpackage.

``docker/full-stack/`` is not itself a Python package, but each runner inside
it is. The runtime image copies the runner packages onto ``/srv``; this
package adds that directory to ``sys.path`` once per session so
``self_improver`` (and its nested ``self_improver.advisor`` subpackage)
imports the same way it does in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FULL_STACK_DIR = Path(__file__).resolve().parent.parent.parent / "docker" / "full-stack"
if str(_FULL_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_STACK_DIR))
