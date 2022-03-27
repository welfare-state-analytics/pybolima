from __future__ import annotations
from dataclasses import dataclass

import os
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from .foss.pos_tags import PoS_Tag_Scheme, PoS_TAGS_SCHEMES
from .foss.stopwords import STOPWORDS
from .interface import CompressType, TaggedIssue
from .utility import replace_extension, store_str, trim_series_type

jj = os.path.join


@dataclass
class DispatchOptions:
    compress_type: Literal['feather', 'csv', 'gzip', 'bz2', 'lzma'] = 'feather'
    to_lower: bool = True
    skip_text: bool = True
    skip_stopwords: bool = False
    skip_puncts: bool = True
    skip_lemma: bool = False


class TaggedFramePerGroupDispatcher:
    def __init__(self, *, target: str, opts: DispatchOptions):
        """Dispatches text blocks to target zink.

        Args:
            target (str): Target filename or folder.
            opts ([str]): Target compress format.
        """
        self.target: str = target
        self.document_id: int = 0
        self.issue_indexes: list[pd.DataFrame] = []
        self.opts: DispatchOptions = opts

    def __enter__(self) -> "TaggedFramePerGroupDispatcher":
        self.open_target(self.target)
        return self

    def __exit__(self, _type, _value, _traceback):  # pylint: disable=unused-argument
        self.close_target()
        return False

    def open_target(self, target: Any) -> None:
        os.makedirs(target, exist_ok=True)

    def close_target(self) -> None:
        self.dispatch_index()

    def dispatch(self, tagged_issue: TaggedIssue) -> None:
        tagged_frame: pd.DataFrame = self.process(tagged_issue)
        filename: str = jj(self.target, tagged_issue.filename)
        self.store(filename=filename, data=tagged_frame)
        self.dispatch_index_item(tagged_issue)

    def dispatch_index_item(self, tagged_issue: TaggedIssue) -> None:
        """Default one document per group"""
        tagged_issue.document_index["document_id"] += self.document_id
        tagged_issue.tagged_frame['document_id'] += self.document_id
        self.document_id += len(tagged_issue.document_index)
        self.issue_indexes.append(tagged_issue.document_index)

    def dispatch_index(self) -> None:
        """Write index of documents to disk."""

        if len(self.issue_indexes) == 0:
            return

        di: pd.DataFrame = pd.concat(self.issue_indexes)
        di.rename({'num_tokens': 'n_tokens'}, inplace=True, errors='ignore')

        di['year'] = trim_series_type(di.year)
        di['n_tokens'] = trim_series_type(di.n_tokens)
        di['document_id'] = trim_series_type(di.document_id)

        self.store(filename=jj(self.target, 'document_index.csv'), data=di)

    def store(self, filename: str, data: str | pd.DataFrame) -> None:
        """Store text to file."""

        if not os.path.split(filename)[0]:
            filename = jj(self.target, f"{filename}")

        if isinstance(data, pd.DataFrame):

            if self.opts.compress_type == 'feather':
                data.to_feather(replace_extension(filename, 'feather'))
                return

            data = data.to_csv(sep='\t')

        if isinstance(data, str):
            store_str(filename=filename, text=data, compress_type=self.opts.compress_type)

    def process(self, item: TaggedIssue) -> pd.DataFrame:

        pads: set = {'MID', 'MAD', 'PAD'}

        tagged_frame: pd.DataFrame = item.tagged_frame.copy()
        tagged_frame['document_id'] += self.document_id

        if self.opts.to_lower:
            tagged_frame["token"] = tagged_frame["token"].str.lower()
            tagged_frame["lemma"] = tagged_frame["lemma"].str.lower()

        drop_columns: list[str] = []

        if 'xpos' in tagged_frame.columns:
            drop_columns.append('xpos')

        if self.opts.skip_stopwords and self.opts.skip_puncts:
            tagged_frame = tagged_frame[
                ~(tagged_frame["token"].str.lower().isin(STOPWORDS) | tagged_frame["pos"].isin(pads))
            ]
        else:
            if self.opts.skip_stopwords:
                tagged_frame = tagged_frame[~tagged_frame["token"].str.lower().isin(STOPWORDS)]
            if self.opts.skip_puncts:
                tagged_frame = tagged_frame[~tagged_frame["pos"].isin(pads)]

        if self.opts.skip_text:
            drop_columns.append('token')
        elif self.opts.to_lower:
            tagged_frame['token'] = tagged_frame['token'].str.lower()

        if self.opts.skip_lemma:
            drop_columns.append('lemma')
        elif self.opts.to_lower:
            tagged_frame['lemma'] = tagged_frame['lemma'].str.lower().fillna('')
            assert not tagged_frame.lemma.isna().any(), "YOU SHALL UPDATE LEMMA FROM TEXT"

        tagged_frame = tagged_frame.drop(columns=drop_columns)
        tagged_frame = tagged_frame.reset_index(drop=True)
        return tagged_frame


class IdTaggedFramePerGroupDispatcher(TaggedFramePerGroupDispatcher):
    def __init__(self, target: str, opts: DispatchOptions):
        super().__init__(target=target, opts=opts)
        self.token2id: defaultdict = defaultdict()
        self.tfs: defaultdict = defaultdict()
        self.token2id.default_factory = self.token2id.__len__
        self.pos_schema: PoS_Tag_Scheme = PoS_TAGS_SCHEMES.SUC

    def process(self, item: TaggedIssue) -> pd.DataFrame:
        tagged_frame: pd.DataFrame = super().process(item)
        fg = lambda t: self.token2id[t]
        pg = self.pos_schema.pos_to_id.get

        if not self.opts.skip_text:
            tagged_frame['token_id'] = tagged_frame.token.apply(fg)

        if not self.opts.skip_lemma:
            tagged_frame['lemma_id'] = tagged_frame.lemma.apply(fg)

        tagged_frame['pos_id'] = tagged_frame.pos.apply(pg).astype(np.int8)
        tagged_frame.drop(columns=['lemma', 'token', 'pos'], inplace=True, errors='ignore')
        return tagged_frame

    def dispatch_index(self) -> None:
        super().dispatch_index()
        self.dispatch_vocabulary()

    def dispatch_vocabulary(self) -> None:
        vocabulary: pd.DataFrame = pd.DataFrame(
            data={
                'token': self.token2id.keys(),
                'token_id': self.token2id.values(),
            }
        )
        self.store(filename=jj(self.target, 'token2id.csv'), data=vocabulary)
