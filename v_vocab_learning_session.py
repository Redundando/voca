from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from toml_i18n import TomlI18n

from v_single_vocab import SingleVocabulary
from v_vocabulary_store import VocabularyStore
from v_vocab_learning_unit import VocabLearningUnit


class VocabLearningSession:
    def __init__(
        self,
        source_language: str,
        target_language: str,
        n: int,
        categories: Sequence[str] | None = None,
        vocab_types: Sequence[str] | None = None,
        store: Optional[VocabularyStore] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if source_language == target_language:
            raise ValueError("Source and target language must differ.")

        self.source_language = source_language
        self.target_language = target_language
        self.n = n
        self.categories = list(categories) if categories is not None else None
        self.vocab_types = list(vocab_types) if vocab_types is not None else None
        self.store = store or VocabularyStore(auto_load=True)
        self.api_key = api_key

    async def run(self) -> List[Dict[str, Any]]:
        cards = self._select_cards()
        results: List[Dict[str, Any]] = []

        for src, tgt in cards:
            unit = VocabLearningUnit(src, tgt, api_key=self.api_key)
            result = await unit.run_interaction()
            results.append(
                {
                    "meaning_en": src.meaning_en,
                    "source_language": src.language,
                    "source_word": src.word,
                    "target_language": tgt.language,
                    "target_word": tgt.word,
                    "score": result.get("score"),
                    "hints": result.get("hints"),
                }
            )

        return results

    def _select_cards(self) -> List[Tuple[SingleVocabulary, SingleVocabulary]]:
        all_vocabs = self.store.all()
        relevant = [
            v
            for v in all_vocabs
            if v.language in {self.source_language, self.target_language}
            and self._matches_vocab_types(v)
            and self._matches_categories(v)
        ]

        groups: Dict[Tuple[str, Tuple[str, ...]], Dict[str, SingleVocabulary]] = defaultdict(dict)
        for v in relevant:
            key = (v.meaning_en, tuple(v.vocab_types or []))
            groups[key][v.language] = v

        candidates: List[Tuple[SingleVocabulary, SingleVocabulary]] = []
        for key, lang_map in groups.items():
            src = lang_map.get(self.source_language)
            tgt = lang_map.get(self.target_language)
            if src and tgt:
                candidates.append((src, tgt))

        if not candidates:
            return []

        sample_n = min(self.n, len(candidates))
        if sample_n < len(candidates):
            candidates = random.sample(candidates, k=sample_n)

        return candidates

    def _matches_vocab_types(self, v: SingleVocabulary) -> bool:
        if not self.vocab_types:
            return True
        return tuple(v.vocab_types or []) == tuple(self.vocab_types)

    def _matches_categories(self, v: SingleVocabulary) -> bool:
        if not self.categories:
            return True
        if not v.categories:
            return False
        return any(c in v.categories for c in self.categories)


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")

    async def demo() -> None:
        session = VocabLearningSession(
            source_language="fr",
            target_language="de",
            n=5,
            categories=["basics"],
            vocab_types=["noun"],
        )
        results = await session.run()
        print(results)

    asyncio.run(demo())
