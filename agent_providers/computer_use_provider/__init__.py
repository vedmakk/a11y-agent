# ---------------------------------------------------------------------------
# Expose sub-packages under the names expected by downstream code **before** we
# import anything that may rely on them (namely ``agent.py``).
# ---------------------------------------------------------------------------

import sys as _sys
import importlib as _importlib

# Make this package importable as ``computer_use_provider`` too.
_sys.modules.setdefault("computer_use_provider", _sys.modules[__name__])

# utils must come first because 'computers' submodules import it.
_sys.modules["utils"] = _importlib.import_module(".utils", __name__)

# Now register computers package.
_sys.modules["computers"] = _importlib.import_module(".computers", __name__)

# Public import (after aliases are registered)
from .cua_provider import CuaAgentProvider

__all__ = [
    "CuaAgentProvider",
]

# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------
# Some legacy sub-modules use absolute imports like ``from computers import …``
# or ``from utils import …`` assuming that *computer_use_provider* is installed
# as a top-level package. To keep those modules unchanged we expose our
# sub-packages under the expected top-level names via ``sys.modules``.

import sys as _sys
import importlib as _importlib

# Alias ``computer_use_provider`` (top-level) ➜ this package instance
_sys.modules.setdefault("computer_use_provider", _sys.modules[__name__])

# Alias nested sub-packages if not already present
if "computers" not in _sys.modules:
    _sys.modules["computers"] = _importlib.import_module(".computers", __name__)

if "utils" not in _sys.modules:
    _sys.modules["utils"] = _importlib.import_module(".utils", __name__)