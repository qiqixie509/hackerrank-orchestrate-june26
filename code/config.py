from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str
    llm_model_name: str = "claude-opus-4-8"
    max_tokens: int = 2048

    # data locations
    dataset_dir: str = "dataset"
    claims_csv: str = "dataset/claims.csv"
    sample_csv: str = "dataset/sample_claims.csv"
    user_history_csv: str = "dataset/user_history.csv"
    evidence_csv: str = "dataset/evidence_requirements.csv"

    # outputs
    output_csv: str = "output.csv"
    limit: int | None = None


settings = Settings()