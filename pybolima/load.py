from __future__ import annotations

from typing import Iterable
import pandas as pd


def load_bolima(filename: str) -> pd.DataFrame:

    corpus: pd.DataFrame = pd.read_parquet(filename)
    corpus['document_name'] = corpus['title'] + "_" + corpus.page.astype(str)
    corpus['issue_name'] = corpus['title']
    corpus['document_id'] = corpus.index
    corpus['year'] = corpus.title.str.split("-").str[1].str[:4].astype(int)

    return corpus


def issue_reader(source: str | pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    corpus: pd.DataFrame = source if isinstance(source, pd.DataFrame) else load_bolima(filename=source)
    titles: list[str] = sorted(corpus['title'].unique())
    for title in titles:
        pages: pd.DataFrame = corpus[corpus['title'] == title]
        yield (title, pages)
