"""The single embedding model CodeSeek indexes and searches with.

Previously this compared two local models (general-purpose vs. code-specific)
since that comparison was itself a project goal. Now that generation moved to
OpenAI, embeddings moved there too (Anthropic has no embeddings endpoint) --
and OpenAI has no code-specialized embedding model, so pairing two generic
OpenAI models under "general"/"code" labels would be a fake comparison, not a
real one. One model, one collection."""

from codeseek.embedding.openai_embedder import OpenAIEmbedder

OPENAI = "openai"


def build_default_embedders() -> dict[str, OpenAIEmbedder]:
    return {OPENAI: OpenAIEmbedder()}
