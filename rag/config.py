from dataclasses import dataclass, field
import os


@dataclass(slots=True)
class RAGConfig:
    data_path: str = "./knowledgeBase.json"
    index_path: str = "./knowledge.index"
    embeddings_path: str = "./knowledge_embeddings.npy"
    inverted_index_path: str = "./inverted_index.json"
    embedding_model_name: str = "shibing624/text2vec-base-chinese"
    default_topk: int = 5
    fusion_k: int = 60
    semantic_candidate_multiplier: int = 2
    keyword_candidate_multiplier: int = 2
    grep_candidate_multiplier: int = 2
    retrieval_pool_size: int = 8
    rerank_pool_size: int = 4
    max_rounds: int = 3
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )
    openai_model: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL")
    )
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "60"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    cohere_api_key: str | None = field(
        default_factory=lambda: os.getenv("COHERE_API_KEY")
    )
    cohere_api_base: str = field(
        default_factory=lambda: os.getenv(
            "COHERE_RERANK_URL",
            "https://api.cohere.com/v2/rerank",
        )
    )
    cohere_model: str = field(
        default_factory=lambda: os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    )
