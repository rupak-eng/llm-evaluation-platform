"""Configuration via environment variables. No secrets in code."""

from __future__ import annotations

import os

from pydantic import BaseModel


class Settings(BaseModel):
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./llmeval.db")
    db_schema: str = os.environ.get("LLMEVAL_DB_SCHEMA", "")
    judge_provider: str = os.environ.get(
        "JUDGE_PROVIDER", "stub"
    )  # stub | groq | openai_compat | deepeval
    judge_model: str = os.environ.get("JUDGE_MODEL", "")
    eval_seed: int = int(os.environ.get("EVAL_SEED", "42"))
    regression_max_drop: float = float(os.environ.get("REGRESSION_MAX_DROP", "0.15"))
    regression_min_kappa: float = float(os.environ.get("REGRESSION_MIN_KAPPA", "0.40"))
    api_host: str = os.environ.get("API_HOST", "127.0.0.1")
    api_port: int = int(os.environ.get("API_PORT", "8020"))

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


settings = Settings()
