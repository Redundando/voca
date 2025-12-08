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
from single_vocabulary import SingleVocabulary

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

    async def pick_random_vocabularies(self, n=5, categories: list[str] | None = None):
        if categories is None:
            subset = self.vocabulary()
        else:
            subset = self.vocabulary()[self.vocabulary()[settings.CATEGORY_COL].isin(categories)]
        subset = subset.sample(n=n)
        vocab_dataframes = []
        for i in range(len(subset)):
            vocab_dataframes.append(subset.iloc[[i]])
        tasks=[]
        result: list[SingleVocabulary] = []
        for df in vocab_dataframes:
            vocab = SingleVocabulary(df = df)
            result.append(vocab)
            tasks+=vocab.pre_process_tasks()
        await asyncio.gather(*tasks)



async def main():
    v = Vocabulary()
    #print(        await v.create_sentence_with_llm())
    await(v.pick_random_vocabularies(categories=["Grundlagen"]))
    #print(vocabs)




if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")
    asyncio.run(main())
