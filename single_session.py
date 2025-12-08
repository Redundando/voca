import asyncio
from datetime import datetime

from cacherator import JSONCache, Cached
from logorator import Logger
from toml_i18n import TomlI18n

from vocabulary import Vocabulary


class SingleSession(JSONCache):

    def __init__(self):
        self.category: str | None = None
        self.date = datetime.now()
        self.category_idx = 0
        self.num_words = 10
        super().__init__(data_id=str(self.date), directory="data/session")

    @Cached(clear_cache=True)
    def vocabulary(self):
        return Vocabulary()

    def list_categories(self):
        for idx, item in enumerate(self.vocabulary().categories(), start=1):
            print(f"{idx:>{3}}: {item}")

    def select_category(self):
        print("Select category\n")
        self.list_categories()
        while True:
            choice = input(f"Choose category [1-{len(self.vocabulary().categories())}]: ").strip()

            if not choice.isdigit():
                print("Enter number.")
                continue

            idx = int(choice)
            if 1 <= idx <= len(self.vocabulary().categories()):
                self.category_idx = idx
                self.category = self.vocabulary().categories()[idx - 1]
                return self.vocabulary().categories()[idx - 1]

            print(f"Number be between 1 and {len(self.vocabulary().categories())}.")

    async def run(self):
        self.select_category()
        await self.vocabulary().pick_random_vocabularies(n=self.num_words, categories=[self.category])
        self.vocabulary().performance_tab().start_background_write(interval=60)
        print(f"Category {self.category}; {self.num_words} words.\n")
        for sv in self.vocabulary().vocab_list:
            await sv.ask_and_rate()
        self.vocabulary().performance_tab().write_data()
        self.vocabulary().align_all()

async def main():
    Logger.set_silent(silent=True)
    ss = SingleSession()
    await ss.run()

if __name__ == "__main__":
    TomlI18n.initialize(locale="en", fallback_locale="en", directory="i18n")
    asyncio.run(main())
