"""Compatibility shim.

The FastAPI app now lives in app_server.main.
Keep this module so existing `app:app` imports continue to work.
"""

from app_server.main import app

__all__ = ["app"]
