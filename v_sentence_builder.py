from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from smartllm import AsyncLLM
from toml_i18n import TomlI18n, i18n

from v_single_vocab import SingleVocabulary


class SentenceBuilder:
    BASE: str = "openai"
    MODEL: str = "gpt-5-mini"
    REASONING_EFFORT: str = "medium"

    SCHEMA_PATH: Path = Path(__file__).parent / "i18n"
    SCHEMA_FILE: str = "vocabularies.yaml"
    PROMPT_KEY: str = "sentence_builder.examples"

    def __init__(
        self,
        vocab: SingleVocabulary,
        n: int = 3,
        api_key: Optional[str] = None,
    ) -> None:
        self.vocab = vocab
        self.n = n

        self.api_key: str | None = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set and no api_key provided.")

        schema_path = self.SCHEMA_PATH / self.SCHEMA_FILE
        with open(schema_path, "r", encoding="utf-8") as f:
            self.json_schema = yaml.safe_load(f)

    async def generate(self) -> List[SingleVocabulary]:
        raw_response = await self._call_llm_for_sentences()
        raw_items = self._parse_llm_response(raw_response)
        sentence_objects = self._build_sentence_objects(raw_items)
        return sentence_objects

    async def _call_llm_for_sentences(self) -> Any:
        prompt = self._build_prompt()

        llm = AsyncLLM(
            base=self.BASE,
            model=self.MODEL,
            api_key=self.api_key,
            prompt=prompt,
            json_schema=self.json_schema,
            reasoning_effort=self.REASONING_EFFORT,
        )

        await llm.execute()
        return llm.response

    def _build_prompt(self) -> str:
        prompt = i18n(
            self.PROMPT_KEY,
            n=self.n,
            meaning_en=self.vocab.meaning_en,
            language=self.vocab.language,
            word=self.vocab.word,
            vocab_types=self.vocab.vocab_types,
        )
        return prompt

    def _parse_llm_response(self, raw_response: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_response, dict):
            raise ValueError(f"Unexpected LLM response type: {type(raw_response)!r}")

        items = raw_response.get("vocabulary")
        if not isinstance(items, list):
            raise ValueError("LLM response does not contain 'vocabulary' as a list")

        cleaned: List[Dict[str, Any]] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            meaning_en = item.get("meaning_en")
            language = item.get("language")
            word = item.get("word")
            categories = item.get("categories") or []
            vocab_types = item.get("vocab_types") or []

            if not isinstance(meaning_en, str) or not isinstance(word, str):
                continue

            cleaned.append(
                {
                    "meaning_en": meaning_en.strip(),
                    "language": language,
                    "word": word.strip(),
                    "categories": list(categories),
                    "vocab_types": list(vocab_types),
                }
            )

        return cleaned

    def _build_sentence_objects(
            self,
            items: List[Dict[str, Any]],
    ) -> List[SingleVocabulary]:
        sentence_objects: List[SingleVocabulary] = []

        base_vocab = self.vocab
        base_categories = list(base_vocab.categories or [])

        for item in items:
            meaning_en: str = item["meaning_en"]
            language: str = item.get("language") or base_vocab.language
            word: str = item["word"]

            item_categories = item.get("categories") or base_categories[:]
            if "example_sentence" not in item_categories:
                item_categories.append("example_sentence")

            item_vocab_types = item.get("vocab_types") or ["phrase"]

            sentence_vocab = SingleVocabulary(
                meaning_en=meaning_en,
                language=language,
                word=word,
                categories=item_categories,
                vocab_types=item_vocab_types,
            )

            # link base vocab -> sentence
            if not any(
                    li.get("data_id") == sentence_vocab.data_id
                    and li.get("relation") == "example_sentence"
                    for li in base_vocab.linked_items
            ):
                base_vocab.linked_items.append(
                    {
                        "data_id" : sentence_vocab.data_id,
                        "relation": "example_sentence",
                    }
                )

            # link sentence -> base vocab
            if not any(
                    li.get("data_id") == base_vocab.data_id
                    and li.get("relation") == "example_for"
                    for li in sentence_vocab.linked_items
            ):
                sentence_vocab.linked_items.append(
                    {
                        "data_id" : base_vocab.data_id,
                        "relation": "example_for",
                    }
                )

            sentence_objects.append(sentence_vocab)

        return sentence_objects


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")

    async def demo() -> None:
        base = SingleVocabulary(
            meaning_en="cherry",
            language="fr",
            word="(la) cerise",
            categories=["basics"],
            vocab_types=["noun"],
        )

        builder = SentenceBuilder(base, n=3)
        sentences = await builder.generate()
        for s in sentences:
            print(repr(s))

    asyncio.run(demo())
