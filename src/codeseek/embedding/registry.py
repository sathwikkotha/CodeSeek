"""The two embedding models under comparison in this project: a general-purpose
sentence embedder and a code-specific one. Both run locally -- no API key,
no per-call cost -- so the eval harness can measure which one actually earns
its keep for code retrieval."""

from codeseek.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder

GENERAL = "general"
CODE = "code"


def build_default_embedders() -> dict[str, SentenceTransformerEmbedder]:
    return {
        GENERAL: SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", dimensions=384),
        # jinaai/jina-embeddings-v2-base-code was tried first but its custom remote
        # code imports a transformers.pytorch_utils symbol removed in transformers v5 --
        # this model is a standard architecture with no custom code, so it isn't
        # exposed to that breakage.
        CODE: SentenceTransformerEmbedder(
            "flax-sentence-embeddings/st-codesearch-distilroberta-base", dimensions=768
        ),
    }
