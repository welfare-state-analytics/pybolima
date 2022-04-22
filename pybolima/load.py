from __future__ import annotations

from typing import Iterable

import pandas as pd


def read_header(filename: str) -> str:
    with open(filename, "r", encoding='utf-8') as fp:
        return fp.readline()


def load_bolima(filename: str) -> pd.DataFrame:

    corpus: pd.DataFrame

    if filename.endswith("feather"):
        corpus = pd.read_feather(filename)

    elif filename.endswith("parquet"):
        corpus: pd.DataFrame = pd.read_parquet(filename)

    else:
        header = read_header(filename)
        sep = '\t' if header.count('\t') > 0 else ','
        corpus = pd.read_csv(filename, sep=sep)

    expected_columns: set[str] = {'title', 'page', 'text'}

    if set(corpus.columns).intersection(expected_columns) != expected_columns:
        raise ValueError(f"column(s) not found: {expected_columns-set(corpus.columns)}")

    if 'document_name' not in corpus.columns:

        corpus['document_name'] = corpus['title'] + "_" + corpus.page.astype(str)
        corpus['issue_name'] = corpus['title']
        corpus['document_id'] = corpus.index
        corpus['year'] = corpus.title.str.split("-").str[1].str[:4].astype(int)

    return corpus


def issue_reader(source: str | pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    corpus: pd.DataFrame = source if isinstance(source, pd.DataFrame) else load_bolima(filename=source)
    titles: list[str] = sorted(corpus['title'].unique())
    for title in tqdm(titles):
        pages: pd.DataFrame = corpus[corpus['title'] == title].copy()
        pages.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
        yield (title, pages)
