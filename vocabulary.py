import asyncio
from typing import Literal

import numpy as np
from cacherator import JSONCache, Cached
from logorator import Logger
from smart_spread import SmartSpread
from toml_i18n import TomlI18n

import performance
import settings
from single_vocabulary import SingleVocabulary


class Vocabulary(JSONCache):

    def __init__(self,
                 key_file=settings.KEY_FILE,
                 sheet_identifier=settings.SHEET_IDENTIFIER,
                 direction: Literal["source", "translation"] = "source"):
        self.vocab_list: list[SingleVocabulary] | None = None
        self.key_file = key_file
        self.sheet_identifier = sheet_identifier
        self.direction = direction
        super().__init__(data_id=self.sheet_identifier, directory="data/vocabulary")

    @Cached(clear_cache=True)
    def sheet(self):
        return SmartSpread(sheet_identifier=self.sheet_identifier, key_file=self.key_file)

    @Cached(clear_cache=True)
    def vocabulary_tab(self):
        return self.sheet().tab(tab_name=settings.VOCAB_TAB_NAME, data_format="DataFrame")

    @Cached(clear_cache=True)
    def performance_tab(self):
        return self.sheet().tab(tab_name=settings.PERFORMANCE_TAB_NAME, data_format="DataFrame")

    @Cached(clear_cache=True)
    def vocabulary(self):
        return self.vocabulary_tab().data

    @Cached(clear_cache=True)
    def categories(self):
        return self.vocabulary()[settings.CATEGORY_COL].unique().tolist()

    async def pick_random_vocabularies(self, n=5, categories: list[str] | None = None):

        def compute_vocab_weights(df, neutral_avg=50.0):
            overall = df["overall rating"].astype(float)
            num = df["num ratings"].astype(float)
            avg = overall / num.replace(0, np.nan)
            avg = avg.fillna(neutral_avg)
            k_eff = num.copy()
            k_eff[k_eff == 0] = 1
            weights = 2.0 ** ((k_eff * (50.0 - avg)) / 50.0)

            return weights

        def sample_vocab(df, n, neutral_avg=50.0, random_state=None):
            if n > len(df):
                raise ValueError("n cannot be larger than the number of rows in df")

            rng = np.random.default_rng(random_state)
            weights = compute_vocab_weights(df, neutral_avg=neutral_avg)
            probs = (weights / weights.sum()).to_numpy()
            indices = df.index.to_numpy()
            chosen_idx = rng.choice(indices, size=n, replace=False, p=probs)

            return df.loc[chosen_idx]

        if categories is None:
            subset = self.vocabulary()
        else:
            subset = self.vocabulary()[self.vocabulary()[settings.CATEGORY_COL].isin(categories)]

        df = sample_vocab(subset, n)

        #print(df.to_string())

        source_lang = settings.SOURCE_LANG
        target_lang = settings.TRANSLATION_LANG

        languages = [source_lang, target_lang]

        self.vocab_list = [SingleVocabulary(languages=languages,
            category=row["category"] if "category" in df.columns else None,
            words=[row["source"], row["translation"]],
            performance_tab=self.performance_tab()) for _, row in df.iterrows()]

    def align_all(self):
        for row in self.vocabulary().to_dict(orient="records"):
            p = performance.Performance(performance_tab=self.performance_tab(),
                                        source=row["source"],
                                        translation=row["translation"])
            self.vocabulary_tab().update_row_by_column_pattern(
                column="source",
                value=row["source"],
                updates={"last check": str(p.last_check()),
                         "overall rating": int(p.overall_rating()),
                         "num ratings": int(p.num_ratings()),}
            )
        self.vocabulary_tab().write_data()


async def main():
    #Logger.set_silent(silent=True)
    v = Vocabulary()
    await v.pick_random_vocabularies(categories=["Grundlagen"])


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")
    asyncio.run(main())
