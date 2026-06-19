from __future__ import annotations

import os
import subprocess
from typing import Dict


def no_window_subprocess_kwargs() -> Dict[str, object]:
    """Return subprocess options that prevent console flashes on Windows."""
    if os.name != "nt":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}
