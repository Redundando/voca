import asyncio
import json
import os
from pathlib import Path

import yaml
from cacherator import JSONCache
from smartllm import AsyncLLM
from toml_i18n import i18n, TomlI18n


class SentenceGenerator(JSONCache):
    def __init__(self, languages: list[str] | None = None, word: str = "", known_vocabularies: list[str] | None = None):
        self.languages = languages
        self.known_vocabularies = known_vocabularies
        self.word = word
        self._sentences: dict | None = None
        super().__init__(data_id=json.dumps({
            "word": self.word,
            "languages": self.languages,
            "known_vocabularies": known_vocabularies}, ensure_ascii=False), directory="data/sentence_generator", )

    async def generate(self) -> dict:
        if self._sentences is not None:
            return self._sentences
        prompt = i18n("vocabulary.generate_sentences",
                      languages=self.languages,
                      word=self.word,
                      known_vocabularies=self.known_vocabularies)

        schema_path = Path(__file__).parent / "i18n" / "sentence.yaml"
        with open(schema_path, "r", encoding="utf-8") as f:
            json_schema = yaml.safe_load(f)

        llm = AsyncLLM(base="openai",
                       model="gpt-5-mini",
                       api_key=os.environ.get("OPENAI_API_KEY"),
                       prompt=prompt,
                       json_schema=json_schema,
                       reasoning_effort="low")
        await llm.execute()
        self._sentences = llm.response
        return self._sentences


async def main():
    sg = SentenceGenerator(languages=["fr", "de"], word="Le vélo", known_vocabularies=None)
    print(await sg.generate())


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")

    asyncio.run(main())
