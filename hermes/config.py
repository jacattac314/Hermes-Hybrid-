from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    router_model: str = "hermes3:70b"
    local_worker_model: str = "llama3.2:3b"

    # Cloud workers
    gemini_api_key: str = ""
    groq_api_key: str = ""
    gemini_model: str = "gemini/gemini-1.5-pro"
    groq_model: str = "groq/llama-3.1-70b-versatile"

    # ChromaDB
    chroma_persist_dir: str = ".chroma"
    memory_collection: str = "hermes_memory"
    memory_top_k: int = 5

    # Routing thresholds
    router_timeout_s: float = 30.0
    dispatcher_timeout_s: float = 60.0


settings = Settings()
