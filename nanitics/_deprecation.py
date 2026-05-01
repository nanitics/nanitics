# PEP 702 `deprecated` decorator sourced from ``typing_extensions`` for 3.11/3.12/3.13
# compatibility. On Python 3.13 ``typing_extensions.deprecated`` is itself a re-export of
# ``warnings.deprecated``, so the observable semantics are identical across the matrix.
from typing_extensions import deprecated

__all__ = ["deprecated"]
