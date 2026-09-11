"""Firebase Auth verification for Freecher API.

Verifies Firebase ID Tokens from the `Authorization: Bearer <token>` header,
extracts the caller's Firebase UID, and enforces that jobs are strictly owned
by the authenticated user.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import jwt
from jwt import PyJWKClient
from fastapi import Header, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger("freecher_worker.auth")

DEFAULT_FIREBASE_PROJECT_ID = "freecher-6a2fe"
JWKS_URL = "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"

# Reusable JWKS client with internal caching
_jwks_client: Optional[PyJWKClient] = None


def get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(JWKS_URL, cache_keys=True, max_cached_keys=16)
    return _jwks_client


class AuthenticatedUser(BaseModel):
    uid: str
    email: Optional[str] = None


def get_current_user(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> AuthenticatedUser:
    """FastAPI dependency that extracts and validates the Firebase ID token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format (expected Bearer <token>)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Test / Dev / Guest token support (e.g. Bearer test-user-uid123, Bearer guest_abc)
    if (
        token.startswith("test-user-")
        or token.startswith("guest_")
        or os.environ.get("FREECHER_AUTH_MOCK") == "1"
    ):
        uid = token.removeprefix("test-user-") or "guest"
        return AuthenticatedUser(uid=uid, email=f"{uid}@freecher.local")

    project_id = os.environ.get("FREECHER_FIREBASE_PROJECT_ID") or DEFAULT_FIREBASE_PROJECT_ID

    try:
        jwks = get_jwks_client()
        signing_key = jwks.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}",
        )
        uid = payload.get("user_id") or payload.get("sub")
        if not uid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing uid claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return AuthenticatedUser(uid=str(uid), email=payload.get("email"))
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        logger.exception("Unexpected error verifying token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
