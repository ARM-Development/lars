import sys
from unittest.mock import MagicMock

# Stub out optional dependencies that may not be available in all environments
# (e.g. asksageclient is only available on the ANL network)
for _mod in ("asksageclient",):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
