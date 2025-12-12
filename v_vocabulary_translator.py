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
from logorator import Logger

class VocabularyTranslator:
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
    async def translate_batch(
        self,
        source_vocabs: Sequence[SingleVocabulary],
        target_language: str,
        batch_size: int = 20,
    ) -> List[SingleVocabulary]:
        if not source_vocabs:
            return []

        results: List[SingleVocabulary] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()

        total = len(source_vocabs)
        idx = 0

        while idx < total:
            batch = source_vocabs[idx : idx + batch_size]
            idx += batch_size

            raw_response = await self._call_llm_for_translations(
                source_vocabs=batch,
                target_language=target_language,
            )

            raw_items = self._parse_llm_response(raw_response)
            raw_items = self._deduplicate_items(raw_items, language=target_language)

            batch_vocabs_all = self._build_vocabulary_objects(
                items=raw_items,
                default_language=target_language,
            )

            new_vocabs: List[SingleVocabulary] = []
            for v in batch_vocabs_all:
                key = (v.language, v.meaning_en, tuple(v.vocab_types or []))
                if key in seen:
                    continue
                seen.add(key)
                new_vocabs.append(v)

            if not new_vocabs and batch:
                # If a batch produced nothing new, just continue;
                # we don't want an infinite loop, but here we advance idx anyway.
                continue

            results.extend(new_vocabs)

        return results

    @Logger()
    async def _call_llm_for_translations(
        self,
        source_vocabs: Sequence[SingleVocabulary],
        target_language: str,
    ) -> Any:
        prompt = self._build_prompt(
            source_vocabs=source_vocabs,
            target_language=target_language,
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
            source_vocabs: Sequence[SingleVocabulary],
            target_language: str,
    ) -> str:
        # Very simple mode switch: if the first vocab is a phrase,
        # we use the sentence-translation prompt.
        first = source_vocabs[0]
        is_phrase = "phrase" in (first.vocab_types or [])

        i18n_key = (
            "vocabulary_translator.batch_phrases"
            if is_phrase
            else "vocabulary_translator.batch"
        )

        items_payload: List[Dict[str, Any]] = []
        for v in source_vocabs:
            items_payload.append(
                {
                    "meaning_en"     : v.meaning_en,
                    "source_language": v.language,
                    "source_word"    : v.word,
                    "categories"     : v.categories,
                    "vocab_types"    : v.vocab_types,
                }
            )

        prompt = i18n(
            i18n_key,
            n=len(source_vocabs),
            target_language=target_language,
            items=json.dumps(items_payload, ensure_ascii=False),
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
    ) -> List[SingleVocabulary]:
        vocabs: List[SingleVocabulary] = []

        for item in items:
            meaning_en: str = item["meaning_en"]
            language: str = item.get("language") or default_language
            word: str = item["word"]

            item_categories = item.get("categories") or []
            item_vocab_types = item.get("vocab_types") or []

            vocab = SingleVocabulary(
                meaning_en=meaning_en,
                language=language,
                word=word,
                categories=item_categories,
                vocab_types=item_vocab_types,
            )
            vocabs.append(vocab)

        return vocabs


async def main() -> None:
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")

    from v_vocabulary_store import VocabularyStore

    store = VocabularyStore(auto_load=True)
    fr_vocabs = store.by_language("fr")

    translator = VocabularyTranslator()
    de_vocabs = await translator.translate_batch(
        source_vocabs=fr_vocabs,
        target_language="de",
        batch_size=10,
    )

    for v in de_vocabs:
        print(repr(v))


if __name__ == "__main__":
    asyncio.run(main())
