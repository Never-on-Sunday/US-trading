"""Secrets and settings loaded from the environment / project-root .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    alpaca_api_key: SecretStr
    alpaca_secret_key: SecretStr
    alpaca_paper: bool = True
    binance_api_key: SecretStr
    binance_secret_key: SecretStr


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()
