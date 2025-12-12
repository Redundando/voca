from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from v_single_vocab import SingleVocabulary


class VocabularyStore:
    def __init__(self, directory: str | Path = "data/vocabulary", auto_load: bool = True) -> None:
        self.directory = Path(directory)
        self._vocabs: List[SingleVocabulary] = []
        if auto_load:
            self.load()

    def load(self) -> None:
        self._vocabs.clear()
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("*.json"):
            data_id = path.stem
            vocab = SingleVocabulary(
                data_id=data_id,
                directory=str(self.directory),
            )
            self._vocabs.append(vocab)

    def all(self) -> List[SingleVocabulary]:
        return list(self._vocabs)

    def by_language(self, language: str) -> List[SingleVocabulary]:
        return [v for v in self._vocabs if getattr(v, "language", None) == language]

    def by_vocab_types(self, vocab_types: Sequence[str]) -> List[SingleVocabulary]:
        target = tuple(vocab_types)
        return [
            v
            for v in self._vocabs
            if tuple(getattr(v, "vocab_types", []) or []) == target
        ]

    def by_language_and_types(self, language: str, vocab_types: Sequence[str]) -> List[SingleVocabulary]:
        target = tuple(vocab_types)
        return [
            v
            for v in self._vocabs
            if getattr(v, "language", None) == language
            and tuple(getattr(v, "vocab_types", []) or []) == target
        ]

    def add(self, vocab: SingleVocabulary, save: bool = True) -> None:
        self._vocabs.append(vocab)
        if save:
            vocab.json_cache_save()

    def extend(self, vocabs: Iterable[SingleVocabulary], save: bool = True) -> None:
        for v in vocabs:
            self.add(v, save=save)

    def save_all(self) -> None:
        for v in self._vocabs:
            v.json_cache_save()

    def __len__(self) -> int:
        return len(self._vocabs)

    def __iter__(self):
        return iter(self._vocabs)


if __name__ == "__main__":
    store = VocabularyStore(auto_load=True)
    print(f"Loaded {len(store)} vocab items")
    for v in store.by_language_and_types("fr", ["phrase"]):
        print(v)
