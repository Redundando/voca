import asyncio
from typing import Literal

from cacherator import JSONCache
from logorator import Logger
from smart_spread import SmartTab
from toml_i18n import TomlI18n

import settings
import vocabulary


class Performance(JSONCache):

    def __init__(self,
                 performance_tab: None | SmartTab = None,
                 source="",
                 translation="",
                 direction: Literal["source", "translation"] = "source"):
        self.performance_tab = performance_tab
        self.source = source
        self.translation = translation
        self.direction = direction
        super().__init__(data_id=f"{self.source} - {self.translation} ({self.translation})", directory="data/performance")

    def performance_data(self):
        if self.performance_tab is None:
            return None
        df = self.performance_tab.data
        return df[(df["source"] == self.source) & (df["translation"] == self.translation) & (df["direction"] == self.direction)]

    def overall_rating(self):
        if self.performance_tab is None or len(self.performance_data())==0:
            return 0
        return self.performance_data()["rating"].sum()

    def num_ratings(self):
        if self.performance_tab is None or len(self.performance_data())==0:
            return 0
        return len(self.performance_data()["rating"])

    def last_check(self):
        if self.performance_tab is None or len(self.performance_data())==0:
            return ""
        return (self.performance_data()["date"]).max()



async def main():
    Logger.set_silent(silent=True)
    v = vocabulary.Vocabulary()
    p = Performance(performance_tab=v.performance_tab(),
                    source="cent", translation="100")
    print(p.performance_data())
    print(p.last_check())


if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")
    asyncio.run(main())
