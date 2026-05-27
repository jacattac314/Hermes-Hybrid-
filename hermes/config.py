from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LM Studio / LM Link (OpenAI-compatible, runs on MacBook Studio)
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_router_model: str = "hermes-3-llama-3.1-70b"
    lmstudio_worker_model: str = "hermes-3-llama-3.1-70b"

    # Router backend: "lmstudio" (default) or "ollama"
    router_backend: str = "lmstudio"

    # Ollama (fallback local backend)
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

    # Routing / dispatch
    router_timeout_s: float = 30.0
    dispatcher_timeout_s: float = 60.0
    stream_default: bool = False


settings = Settings()
