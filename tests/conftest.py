import os
from os.path import isdir
from os.path import join as jj

import pandas as pd
from pytest import fixture

from pybolima.interface import TaggedIssue

from . import SAMPLE_CORPUS_FILENAME

if not isdir(jj("tests", "output")):
    os.makedirs(jj("tests", "output"))


@fixture
def bolima_corpus_sample_corpus():
    data: pd.DataFrame = pd.read_csv(SAMPLE_CORPUS_FILENAME, sep='\t', index_col=0)
    return data


@fixture
def tagged_issues() -> list[TaggedIssue]:
    return TaggedIssue.load_all("tests/test_data")
