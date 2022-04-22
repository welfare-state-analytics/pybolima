from __future__ import annotations

import typing as t
from io import StringIO

import pandas as pd
from loguru import logger
from tqdm import tqdm

from pybolima.dispatch import TaggedFramePerGroupDispatcher
from pybolima.load import issue_reader

from .interface import TaggedIssue
from .stanza import ITagger
from .transform import normalize_characters


def tag_issues(
    tagger: ITagger,
    source: str | pd.DataFrame,
    target: str,
    dispatch_cls: t.Type[TaggedFramePerGroupDispatcher],
    dispatch_opts: t.Type[TaggedFramePerGroupDispatcher],
):

    with dispatch_cls(target=target, opts=dispatch_opts) as dispatcher:
        for title, pages in issue_reader(source=source):
            try:
                tagged_issue: TaggedIssue = tag_issue(
                    tagger=tagger, title=title, issue_pages=pages, normalize_chars=True
                )
                dispatcher.dispatch(tagged_issue=tagged_issue)
            except Exception as ex:
                logger.info(f"failed: {title} {ex}")


def tag_issue(
    *,
    tagger: ITagger,
    title: str,
    issue_pages: pd.DataFrame,
    normalize_chars: None | str | bool = None,
) -> TaggedIssue:

    texts = issue_pages['text'].to_list()
    document_index: pd.DataFrame = issue_pages.reset_index()
    document_index.drop(columns="text", inplace=True)

    if normalize_chars is not False:
        texts = [normalize_characters(text) for text in texts]

    tagged_data: list[dict[str, list[str]]] = tagger.tag(texts)
    tagged_pages: list[pd.DataFrame] = []

    for i, tagged_page in enumerate(tagged_data):

        tagged_csv_str: str = tagger.to_csv(tagged_page)
        tagged_page: pd.DataFrame = pd.read_csv(StringIO(tagged_csv_str), sep='\t', quoting=3)

        tagged_page['document_id'] = i
        tagged_page.drop(columns="xpos", inplace=True)
        tagged_pages.append(tagged_page)

    tagged_issue_frame: pd.DataFrame = pd.concat(tagged_pages)

    document_index["n_tokens"] = [d["n_tokens"] for d in tagged_data]
    document_index["n_words"] = [d["n_words"] for d in tagged_data]

    return TaggedIssue(title=title, document_index=document_index, tagged_frame=tagged_issue_frame)
