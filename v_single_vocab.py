from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

from cacherator import JSONCache

import asyncio


class SingleVocabulary(JSONCache):
    def __init__(
        self,
        meaning_en: Optional[str] = None,
        language: Optional[str] = None,              # e.g. "fr", "de"
        vocab_types: Optional[List[str]] = None,     # e.g. ["noun"], ["phrase"]
        *,
        word: Optional[str] = None,                  # e.g. "chien", "Hund"
        categories: Optional[List[str]] = None,
        linked_items: Optional[list[dict[str, str]]] = None,
        data_id: Optional[str] = None,
        directory: Optional[str] = None,
        clear_cache: bool = False,
        ttl: Optional[int | float | timedelta] = 99999,
        logging: bool = False,
    ) -> None:
        # If no data_id is given, we are creating a NEW vocab and need the triple
        if data_id is None:
            if not meaning_en or not language:
                raise ValueError(
                    "meaning_en and language must be provided when data_id is not set."
                )
            if not vocab_types:
                raise ValueError(
                    "vocab_types must be provided when data_id is not set."
                )
            vocab_types = vocab_types or []
            resolved_data_id = f"{language}_{'_'.join(vocab_types)}_{meaning_en}"
        else:
            resolved_data_id = data_id
            # vocab_types may be None when loading by data_id

        super().__init__(
            data_id=resolved_data_id,
            directory=directory or "data/vocabulary",
            clear_cache=clear_cache,
            ttl=ttl,
            logging=logging,
        )

        # Only set attributes if they don't exist yet AND we actually got a value.
        # For cached objects, JSONCache will already have restored them.
        if meaning_en is not None and not hasattr(self, "meaning_en"):
            self.meaning_en = meaning_en
        if language is not None and not hasattr(self, "language"):
            self.language = language
        if word is not None and not hasattr(self, "word"):
            self.word = word
        if not hasattr(self, "categories"):
            self.categories: List[str] = categories or []
        if vocab_types is not None and not hasattr(self, "vocab_types"):
            self.vocab_types: List[str] = vocab_types or []
        if not hasattr(self, "linked_items"):
            self.linked_items: list[dict[str, str]] = linked_items or []

    @property
    def data_id(self) -> str:
        return self._json_cache_data_id

    @property
    def languages(self) -> List[str]:
        return ["en", self.language]

    def load_linked_sentences(self) -> List["SingleVocabulary"]:
        sentences: List[SingleVocabulary] = []
        seen: set[str] = set()
        directory = getattr(self, "_json_cache_directory", None)

        for link in self.linked_items:
            if link.get("relation") != "example_sentence":
                continue
            data_id = link.get("data_id")
            if not data_id or data_id in seen:
                continue
            seen.add(data_id)
            try:
                sentence = SingleVocabulary(
                    data_id=data_id,
                    directory=directory,
                )
                sentences.append(sentence)
            except Exception:
                continue

        return sentences

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"meaning_en={getattr(self, 'meaning_en', None)!r}, "
            f"language={getattr(self, 'language', None)!r}, "
            f"word={getattr(self, 'word', None)!r}, "
            f"categories={getattr(self, 'categories', None)!r}, "
            f"vocab_types={getattr(self, 'vocab_types', None)!r}, "
            f"linked_items={getattr(self, 'linked_items', None)!r})"
        )

    def __str__(self) -> str:
        parts: List[str] = [
            f"'{getattr(self, 'meaning_en', '')}' "
            f"({getattr(self, 'language', '?')}: {getattr(self, 'word', '')})"
        ]
        if getattr(self, "categories", None):
            parts.append(f"[categories: {', '.join(self.categories)}]")
        if getattr(self, "vocab_types", None):
            parts.append(f"[types: {', '.join(self.vocab_types)}]")
        return " ".join(parts)

    def __len__(self) -> int:
        return len(getattr(self, "word", "") or "")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleVocabulary):
            return NotImplemented
        return (
            getattr(self, "meaning_en", None) == getattr(other, "meaning_en", None)
            and getattr(self, "language", None) == getattr(other, "language", None)
            and getattr(self, "word", None) == getattr(other, "word", None)
            and getattr(self, "categories", None) == getattr(other, "categories", None)
            and getattr(self, "vocab_types", None) == getattr(other, "vocab_types", None)
            and getattr(self, "linked_items", None) == getattr(other, "linked_items", None)
        )


async def main() -> None:
    vocab = SingleVocabulary(
        meaning_en="cherry",
        language="fr",
        vocab_types=["noun"],
    )

    print("repr :", repr(vocab))
    print("str  :", str(vocab))
    print("len  :", len(vocab))
    print("langs:", vocab.languages)

    linked = vocab.load_linked_sentences()
    print(linked)


if __name__ == "__main__":
    asyncio.run(main())
