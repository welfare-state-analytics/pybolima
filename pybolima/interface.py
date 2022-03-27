import zipfile
from dataclasses import dataclass
from enum import Enum
from glob import glob
from os.path import basename
from os.path import join as jj

import pandas as pd


@dataclass
class TaggedIssue:
    title: str
    document_index: pd.DataFrame
    tagged_frame: pd.DataFrame

    @property
    def safe_title(self) -> str:
        return self.title.replace(":", "_#")

    @property
    def filename(self) -> str:
        return f"{self.safe_title}.csv"

    @property
    def index_name(self) -> str:
        return f"{self.safe_title}_document_index.csv"

    @staticmethod
    def load(folder: str, title: str) -> "TaggedIssue":
        item: TaggedIssue = TaggedIssue(title=title, tagged_frame=None, document_index=None)
        item.tagged_frame = pd.read_csv(jj(folder, item.filename), sep='\t', index_col=0)
        item.document_index = pd.read_csv(jj(folder, item.index_name), sep='\t', index_col=0)
        return item

    @staticmethod
    def load_all(folder: str) -> list["TaggedIssue"]:
        return [TaggedIssue.load(folder, title) for title in TaggedIssue.find(folder)]

    @staticmethod
    def find(folder: str) -> list[str]:
        filenames: list[str] = sorted(glob(jj(folder, "BLM-*_document_index.csv")))
        return [
            'BLM-' + basename(filename).removeprefix('BLM-').removesuffix('_document_index.csv').replace("_#", ":")
            for filename in filenames
        ]

    def store(self, folder: str):
        self.tagged_frame.to_csv(jj(folder, self.filename), '\t')
        self.document_index.to_csv(jj(folder, self.index_name), '\t')


class CompressType(str, Enum):
    Plain = 'csv'
    Zip = 'zip'
    Gzip = 'gzip'
    Bz2 = 'bz2'
    Lzma = 'lzma'
    Feather = 'feather'

    def to_zipfile_compression(self):
        if self.value == "csv":
            return zipfile.ZIP_STORED
        if self.value == "bz2":
            return zipfile.ZIP_BZIP2
        if self.value == "lzma":
            return zipfile.ZIP_LZMA
        return zipfile.ZIP_DEFLATED

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]
