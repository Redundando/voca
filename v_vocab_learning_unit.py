from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from smartllm import AsyncLLM
from toml_i18n import TomlI18n, i18n

from v_single_vocab import SingleVocabulary


class VocabLearningUnit:
    BASE: str = "openai"
    MODEL: str = "gpt-5-mini"
    REASONING_EFFORT: str = "low"

    SCHEMA_PATH: Path = Path(__file__).parent / "i18n"
    RATING_SCHEMA_FILE: str = "translation_rating.yaml"
    RATING_PROMPT_KEY: str = "vocab_learning_unit.evaluate_translation"

    def __init__(
        self,
        source_vocab: SingleVocabulary,
        target_vocab: SingleVocabulary,
        api_key: Optional[str] = None,
    ) -> None:
        if source_vocab.meaning_en != target_vocab.meaning_en:
            raise ValueError("Source and target vocab must share the same meaning_en")
        if tuple(source_vocab.vocab_types or []) != tuple(target_vocab.vocab_types or []):
            raise ValueError("Source and target vocab must share the same vocab_types")

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.api_key: str | None = api_key
        if not self.api_key:
            from os import environ

            self.api_key = environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set and no api_key provided.")

        schema_path = self.SCHEMA_PATH / self.RATING_SCHEMA_FILE
        with open(schema_path, "r", encoding="utf-8") as f:
            self.rating_schema = yaml.safe_load(f)

        self.last_score: Optional[int] = None
        self.last_feedback: Optional[str] = None

    def question_text(self) -> str:
        return self.source_vocab.word

    async def run_interaction(self) -> Dict[str, Any]:
        from google_tts import speak

        def _spoken_form(text: str) -> str:
            without_parens = re.sub(r"\([^)]*\)", " ", text)
            return " ".join(without_parens.split())


        speak(_spoken_form(self.source_vocab.word), lang=self.source_vocab.language)

        while True:
            print()
            print(f"Translate the word into '{self.target_vocab.language}' or choose an option:")
            print("  (1) Repeat word")
            print("  (2) Use word in a sentence  [not implemented yet]")
            print("  (3) Show word in source language")
            print("  (Enter) I don't know / skip")

            user_input = input("> ").strip()

            if user_input == "1":
                speak(_spoken_form(self.source_vocab.word), lang=self.source_vocab.language)
                continue

            if user_input == "2":
                print("[Using the word in a sentence is not implemented yet.]")
                continue

            if user_input == "3":
                print(f"Source word: {self.source_vocab.word}")
                continue

            if user_input == "":
                result = {"score": 0, "hints": "", "raw": None}
                self.last_score = 0
                self.last_feedback = ""
                return result

            result = await self.evaluate_answer(user_answer=user_input)
            print(f"{self.last_score} {self.last_feedback}")
            return result

    async def evaluate_answer(self, user_answer: str) -> Dict[str, Any]:
        prompt = self._build_rating_prompt(user_answer=user_answer)

        llm = AsyncLLM(
            base=self.BASE,
            model=self.MODEL,
            api_key=self.api_key,
            prompt=prompt,
            json_schema=self.rating_schema,
            reasoning_effort=self.REASONING_EFFORT,
        )

        await llm.execute()
        result = self._parse_rating_response(llm.response)

        self.last_score = result.get("score")
        self.last_feedback = result.get("hints")

        return result

    def _build_rating_prompt(self, user_answer: str) -> str:
        prompt = i18n(
            self.RATING_PROMPT_KEY,
            from_language=self.source_vocab.language,
            to_language=self.target_vocab.language,
            source_word=self.source_vocab.word,
            meaning_en=self.source_vocab.meaning_en,
            user_translation=user_answer,
        )
        return prompt

    def _parse_rating_response(self, raw_response: Any) -> Dict[str, Any]:
        if not isinstance(raw_response, dict):
            raise ValueError(f"Unexpected rating response type: {type(raw_response)!r}")

        score = raw_response.get("score")
        hints = raw_response.get("hints")

        return {
            "score": score,
            "hints": hints,
            "raw": raw_response,
        }


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")

    src = SingleVocabulary(
        meaning_en="book",
        language="fr",
        word="(le) livre",
        categories=["basics"],
        vocab_types=["noun"],
    )
    tgt = SingleVocabulary(
        meaning_en="book",
        language="de",
        word="(das) Buch",
        categories=["basics"],
        vocab_types=["noun"],
    )

    async def demo() -> None:
        unit = VocabLearningUnit(src, tgt)
        result = await unit.run_interaction()
        print(result)

    import asyncio

    asyncio.run(demo())
