"""
Centralized runtime configuration.

Values are loaded from environment variables (and a local `.env` file during
development) via pydantic-settings. This is the ONLY place environment
variables should be read in the backend — no other module should call
`os.environ` directly, so secrets stay auditable in one file.

CRITICAL SECURITY RULE (see ../../SECURITY.md):
Secrets loaded here (e.g. GEMINI_API_KEY) must never be serialized into an
HTTP/WebSocket response sent to the frontend. Only derived, non-secret flags
(e.g. "is a key configured: true/false") may ever cross that boundary.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required
    gemini_api_key: Optional[str] = None

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:4321"

    # Database / cache (required starting Segment 16 — Memory)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None

    # Optional third-party providers (all deferred; system must work without them)
    maps_provider_api_key: Optional[str] = None
    weather_provider_api_key: Optional[str] = None
    external_speech_provider_api_key: Optional[str] = None

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
