import asyncio
import json
import os
import random
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Literal

import yaml
from cacherator import JSONCache, Cached
from smartllm import AsyncLLM
from smart_spread import smart_tab
from toml_i18n import i18n, TomlI18n
from logorator import Logger
import sentence_generator
from google_tts import speak
import performance

class SingleVocabulary(JSONCache):
    def __init__(self,
                 languages: list[str] | None = None,
                 category: str | None = None,
                 words: list[str] | None = None,
                 known_vocabularies: list[str] | None = None,
                 sentences: list[dict] | None = None,
                 direction: Literal["source", "translation"] = "source",
                 vocabulary_tab:smart_tab.SmartTab | None = None,
                 performance_tab: smart_tab.SmartTab | None = None,):
        self.rating_given: None | dict = None
        self.translation_given: None | str = None
        self.languages = languages
        self.category = category
        self.words = words
        self.known_vocabularies = known_vocabularies
        self._sentences = sentences
        self.direction = direction
        self.performance_tab = performance_tab
        self.vocabulary_tab = vocabulary_tab

        super().__init__(data_id=json.dumps({"languages": self.languages, "words": self.words}, ensure_ascii=False),
                         directory="data/single_vocabulary", )

    @Cached()
    def performance(self):
        return performance.Performance(performance_tab=self.performance_tab,source=self.words[0], translation=self.words[1], direction=self.direction)

    async def sentences(self):
        if self._sentences is not None:
            return self._sentences
        sg = sentence_generator.SentenceGenerator(languages=self.languages,
                                                  word=self.words[0],
                                                  known_vocabularies=self.known_vocabularies)
        self._sentences = (await sg.generate()).get("sentences")
        return self._sentences

    async def say_word(self) -> None:
        word = self.words[0] if self.direction == "source" else self.words[1]
        lang = self.languages[0] if self.direction == "source" else self.languages[1]
        if word:
            speak(text=word, lang=lang)

    async def say_sentence(self, sentence_nr: int = 0) -> None:
        sentences = await self.sentences()
        sentence_nr = sentence_nr % len(sentences)
        sentence = sentences[sentence_nr].get("source") if self.direction == "source" else sentences[sentence_nr].get(
            "translation")
        lang = self.languages[0] if self.direction == "source" else self.languages[1]
        if sentence:
            speak(text=sentence, lang=lang)

    async def ask_translation(self) -> str:
        await self.say_word()

        while True:
            user_input = input("Enter translation or (1) repeat word, (2) use in sentence, (3) see written word: ").strip()

            if user_input == "1":
                await self.say_word()
            elif user_input == "2":
                await self.say_sentence(sentence_nr=random.randint(1, len(await self.sentences())))
            elif user_input == "3":
                print(f"\n{self.words[0] if self.direction == "source" else self.words[1]}\n")
            else:
                return user_input

    async def check_translation(self, translation: str = ""):
        if translation.strip() == "":
            return {"rating": -1, "hints": ""}
        prompt = i18n("vocabulary.check_translation_simple",
                      from_language=self.languages[0] if self.direction == "source" else self.languages[1],
                      to_language=self.languages[1] if self.direction == "source" else self.languages[0],
                      word=self.words[0] if self.direction == "source" else self.words[1],
                      translation=translation)
        schema_path = Path(__file__).parent / "i18n" / "check_translation_simple.yaml"
        with open(schema_path, "r", encoding="utf-8") as f:
            json_schema = yaml.safe_load(f)
        llm = AsyncLLM(base="openai",
                       model="gpt-5-nano",
                       reasoning_effort="minimal",
                       api_key=os.environ.get("OPENAI_API_KEY"),
                       prompt=prompt,
                       json_schema=json_schema, )
        await llm.execute()
        return llm.response

    async def rate_translation(self, translation: str = ""):
        if translation.strip() == "":
            return {"rating": -1, "hints": ""}
        prompt = i18n("vocabulary.rate_translation",
                      from_language=self.languages[0] if self.direction == "source" else self.languages[1],
                      to_language=self.languages[1] if self.direction == "source" else self.languages[0],
                      word=self.words[0] if self.direction == "source" else self.words[1],
                      translation=translation)
        schema_path = Path(__file__).parent / "i18n" / "translation_rating.yaml"
        with open(schema_path, "r", encoding="utf-8") as f:
            json_schema = yaml.safe_load(f)
        llm = AsyncLLM(base="openai",
                       model="gpt-5-mini",
                       reasoning_effort="low",
                       api_key=os.environ.get("OPENAI_API_KEY"),
                       prompt=prompt,
                       json_schema=json_schema, )
        await llm.execute()
        return llm.response

    async def ask_and_rate(self):
        Logger.set_silent()
        print("=================================================")
        self.translation_given = await self.ask_translation()
        print("=================================================")
        print(self.words)
        print("=================================================")
        self.rating_given = await self.rate_translation(translation=self.translation_given)
        print(self.rating_given)
        if self._sentences is not None:
            print("=================================================")
            pprint(self._sentences)
        print("\n\n")
        if self.performance_tab:
            self.performance_tab.data.loc[len(self.performance_tab.data)] = {
                "date": str(datetime.now()),
                "source": self.words[0] if self.direction == "source" else self.words[1],
                "translation": self.words[1] if self.direction == "source" else self.words[0],
                "user_input":self.translation_given,
                "rating":self.rating_given.get("rating"),
                "hint":self.rating_given.get("hints"),
                "direction":self.direction,
            }

    async def align(self):
        if self.performance_tab and self.vocabulary_tab:
            self.vocabulary_tab.update_row_by_column_pattern(
                column="source",
                value=self.words[0],
                updates={"last check"    : str(self.performance.last_check()),
                         "overall rating": int(self.performance.overall_rating()),
                         "num ratings"   : int(self.performance.num_ratings()), }
            )

async def main():
    sv = SingleVocabulary(languages=["fr", "de"], words=["Le garçon", "Der Junge"])
    print(await sv.sentences())
    trans = await sv.ask_translation()
    print(await sv.check_translation(translation=trans))
    print(await sv.rate_translation(translation=trans))


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")

    asyncio.run(main())
