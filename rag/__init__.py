from .config import RAGConfig

__all__ = [
    "ToolCallingRAGAgent",
    "ModularRAGOrchestrator",
    "RAGConfig",
    "SelfRAGPipeline",
]


def __getattr__(name: str):
    if name == "ToolCallingRAGAgent":
        from .agent import ToolCallingRAGAgent

        return ToolCallingRAGAgent
    if name == "ModularRAGOrchestrator":
        from .orchestrator import ModularRAGOrchestrator

        return ModularRAGOrchestrator
    if name == "SelfRAGPipeline":
        from .self_rag import SelfRAGPipeline

        return SelfRAGPipeline
    raise AttributeError(f"module 'rag' has no attribute {name!r}")
