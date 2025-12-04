from smart_spread import SmartSpread
from cacherator import JSONCache, Cached
import pandas as pd
from google_tts import speak
import asyncio
import os
from smartllm import AsyncLLM
from toml_i18n import TomlI18n, i18n
import yaml
from pathlib import Path
import settings
import json

class Vocabulary(JSONCache):

    def __init__(self, key_file=settings.KEY_FILE, sheet_identifier=settings.SHEET_IDENTIFIER):
        self.key_file = key_file
        self.sheet_identifier = sheet_identifier
        super().__init__(data_id=self.sheet_identifier, directory="data/vocabulary")

    @Cached(clear_cache=True)
    def sheet(self):
        return SmartSpread(sheet_identifier=self.sheet_identifier, key_file=self.key_file)

    @Cached(clear_cache=True)
    def vocabulary_tab(self):
        return self.sheet().tab(tab_name=settings.VOCAB_TAB_NAME, data_format="DataFrame")

    @Cached(clear_cache=True)
    def vocabulary(self):
        return self.vocabulary_tab().data

    @Cached(clear_cache=True)
    def categories(self):
        return self.vocabulary()[settings.CATEGORY_COL].unique().tolist()

    def pick_random_vocabularies(self, n=10, categories: list[str] | None = None):
        if categories is None:
            subset = self.vocabulary()
        else:
            subset = self.vocabulary()[self.vocabulary()[settings.CATEGORY_COL].isin(categories)]
        return subset.sample(n=n)

    def say_vocabulary(self, vocabulary: pd.DataFrame, language: str = "fr"):
        vocab = vocabulary.iloc[0].to_dict()
        speak(text=vocab.get(language, ""), lang=language)


class SingleVocabulary(JSONCache):
    def __init__(self, df: pd.DataFrame):
        self.languages = [df.columns[0], df.columns[1]]
        self.category = df[settings.CATEGORY_COL].iloc[0]
        self.words = {}
        for l in self.languages:
            self.words[l] =  df[l].iloc[0]
        self.sentences = {}
        super().__init__(data_id=json.dumps(self.words, ensure_ascii=False), directory="data/single_vocabulary")



    async def sentence(self, lang=""):
        if self.sentences.get(lang) is not None:
            return self.sentences[lang]
        if lang not in self.languages:
            return ""
        prompt = i18n("vocabulary.generate_sentence", word=self.words.get(lang), language=lang)
        with open(str(Path(__file__).parent / "i18n/sentence.yaml"), "r") as f:
            json_schema = yaml.safe_load(f)

        llm = AsyncLLM(base="openai", model="gpt-5", api_key=os.environ.get("OPENAI_API_KEY"), prompt=prompt,
                       json_schema=json_schema)
        await llm.execute()
        self.sentences[lang] = llm.response.get("sentence")
        return self.sentences[lang]

    async def say_sentence(self, lang=""):
        speak(text=await self.sentence(lang), lang=lang)

    async def say_word(self, lang=""):
        speak(text=self.words.get(lang), lang=lang)

    async def say_vocab(self, lang=""):
        await self.say_word(lang=lang)
        #await asyncio.sleep(0.02)
        await self.say_sentence(lang=lang)
        #await asyncio.sleep(0.02)
        await self.say_word(lang=lang)

async def main():
    v = Vocabulary()
    #print(        await v.create_sentence_with_llm())
    vocabs = v.pick_random_vocabularies()
    print(vocabs)
    vv = SingleVocabulary(vocabs)
    print(await vv.sentence(lang="fr"))
    print(await vv.sentence(lang="de"))
    await vv.say_vocab(lang="fr")
    await vv.say_vocab(lang="de")




if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")
    asyncio.run(main())
