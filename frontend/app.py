"""Compatibility shim.

The FastAPI app now lives in backend.main.
Keep this module so existing `app:app` imports continue to work.
"""

from backend.main import app

__all__ = ["app"]
