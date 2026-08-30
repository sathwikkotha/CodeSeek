"""HyDE (Hypothetical Document Embeddings): ask a small, cheap model to write
a short hypothetical answer/snippet for the query, and embed *that* instead of
the raw question -- closes the vocabulary gap between a natural-language
question ("how do I set a default value for a flag") and code-shaped chunks
("default: Any = None") that share almost no words with it.

Deliberately narrow in scope: this only changes what gets embedded for the
*vector* leg of hybrid search. The keyword/BM25 leg (retrieval/hybrid.py)
keeps searching the user's original literal words, since a hypothetical
snippet may use different variable names/terms than what the user actually
typed -- expanding the keyword query too would risk losing exact-term
matches the literal query would have caught."""

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.4-nano"

_SYSTEM_PROMPT = (
    "You help a code search engine. Given a natural-language question about a "
    "codebase, write a short (3-6 lines) hypothetical snippet of code or "
    "technical explanation that a correct answer might contain -- as if it "
    "were pulled directly from the relevant source or documentation. Output "
    "only the hypothetical snippet/explanation, nothing else: no preamble, "
    "no reasoning, no markdown fences."
)


class HydeQueryExpander:
    def __init__(self, model: str = DEFAULT_MODEL, client: OpenAI | None = None):
        self.model = model
        self._client = client or OpenAI()

    def expand(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_completion_tokens=200,
        )
        content = response.choices[0].message.content
        # A HyDE failure (empty completion, content filter) should degrade to
        # the plain query, not to embedding an empty string.
        return content.strip() if content else query
