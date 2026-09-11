"""HTTP API for the Freecher job pipeline."""
from freecher_worker.api.app import create_app

__all__ = ["create_app"]
