from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from smartllm import AsyncLLM
from toml_i18n import TomlI18n, i18n

from v_single_vocab import SingleVocabulary
from v_vocabulary_store import VocabularyStore
from logorator import Logger

class VocabularyBuilder:
    BASE: str = "openai"
    MODEL: str = "gpt-5"
    REASONING_EFFORT: str = "low"

    SCHEMA_PATH: Path = Path(__file__).parent / "i18n"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str | None = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set and no api_key provided.")
        schema_path = self.SCHEMA_PATH / "vocabularies.yaml"
        with open(schema_path, "r", encoding="utf-8") as f:
            self.json_schema = yaml.safe_load(f)

    @Logger()
    async def generate(
        self,
        n: int,
        language: str,
        categories: Sequence[str] | None = None,
        vocab_types: Sequence[str] | None = None,
        existing_vocab: Sequence[SingleVocabulary] | None = None,
        batch_size: int = 20,
    ) -> List[SingleVocabulary]:
        results: List[SingleVocabulary] = []

        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        if existing_vocab:
            for v in existing_vocab:
                key = (v.language, v.meaning_en, tuple(v.vocab_types or []))
                seen.add(key)

        remaining = n

        while remaining > 0:
            this_batch_n = min(batch_size, remaining)

            combined_existing: List[SingleVocabulary] = []
            if existing_vocab:
                combined_existing.extend(existing_vocab)
            combined_existing.extend(results)

            raw_response = await self._call_llm_for_vocabulary(
                n=this_batch_n,
                language=language,
                categories=categories,
                vocab_types=vocab_types,
                existing_vocab=combined_existing or None,
            )

            raw_items = self._parse_llm_response(raw_response)
            raw_items = self._deduplicate_items(raw_items, language=language)

            batch_vocabs_all = self._build_vocabulary_objects(
                items=raw_items,
                default_language=language,
                categories=categories,
                vocab_types=vocab_types,
            )

            new_vocabs: List[SingleVocabulary] = []
            for v in batch_vocabs_all:
                key = (v.language, v.meaning_en, tuple(v.vocab_types or []))
                if key in seen:
                    continue
                seen.add(key)
                new_vocabs.append(v)

            if not new_vocabs:
                break

            results.extend(new_vocabs)
            remaining = n - len(results)

        return results[:n]

    @Logger()
    async def _call_llm_for_vocabulary(
        self,
        n: int,
        language: str,
        categories: Sequence[str] | None = None,
        vocab_types: Sequence[str] | None = None,
        existing_vocab: Sequence[SingleVocabulary] | None = None,
    ) -> Any:
        prompt = self._build_prompt(
            n=n,
            language=language,
            categories=categories,
            vocab_types=vocab_types,
            existing_vocab=existing_vocab,
        )

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

    def _build_prompt(
        self,
        n: int,
        language: str,
        categories: Sequence[str] | None = None,
        vocab_types: Sequence[str] | None = None,
        existing_vocab: Sequence[SingleVocabulary] | None = None,
    ) -> str:
        existing_meanings_en: List[Dict[str, Any]] = []
        if existing_vocab:
            existing_meanings_en = [
                {"meaning": v.meaning_en, "types": v.vocab_types} for v in existing_vocab
            ]

        if categories == ["basics"] and vocab_types == ["noun"]:
            prompt = i18n(
                "vocabulary_builder.starter_nouns",
                n=n,
                language=language,
                existing_meanings_en=json.dumps(
                    existing_meanings_en, ensure_ascii=False
                ),
            )
            return prompt

        raise NotImplementedError(
            f"Prompt for {categories} / {vocab_types} is not implemented."
        )

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

    def _deduplicate_items(
        self,
        items: List[Dict[str, Any]],
        language: str,
    ) -> List[Dict[str, Any]]:
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        result: List[Dict[str, Any]] = []
        for item in items:
            meaning_en = item.get("meaning_en")
            lang = item.get("language") or language
            vocab_types = item.get("vocab_types") or []
            key = (lang, meaning_en, tuple(vocab_types))
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _build_vocabulary_objects(
        self,
        items: List[Dict[str, Any]],
        default_language: str,
        categories: Sequence[str] | None,
        vocab_types: Sequence[str] | None,
    ) -> List[SingleVocabulary]:
        vocab_objects: List[SingleVocabulary] = []

        for item in items:
            meaning_en: str = item["meaning_en"]
            language: str = item.get("language") or default_language
            word: str = item["word"]

            item_categories = item.get("categories") or list(categories or [])
            item_vocab_types = item.get("vocab_types") or list(vocab_types or [])

            vocab = SingleVocabulary(
                meaning_en=meaning_en,
                language=language,
                word=word,
                categories=item_categories,
                vocab_types=item_vocab_types,
            )
            vocab_objects.append(vocab)

        return vocab_objects


async def main() -> None:
    store = VocabularyStore(auto_load=True)
    french_vocab = store.by_language("fr")
    vb = VocabularyBuilder()
    vocabs = await vb.generate(
        n=20,
        language="fr",
        vocab_types=["noun"],
        categories=["basics"],
        existing_vocab=french_vocab,
        batch_size = 10
    )
    for v in vocabs:
        print(repr(v))


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")
    asyncio.run(main())
