import os
from typing import Iterable

import pandas as pd
import pytest
from pytest import fixture

from pybolima.dispatch import TaggedIssue
from pybolima.load import issue_reader, load_bolima
from pybolima.stanza import ITagger, StanzaTagger
from pybolima.tagger import tag_issue
from pybolima.utility import pretokenize

from . import DATA_FILENAME, MODEL_ROOT, SAMPLE_CORPUS_FILENAME, TEST_DOCUMENTS

# pylint: disable=redefined-outer-name

if not os.path.isdir(MODEL_ROOT):
    pytest.skip(f"Skipping Stanza tests since model path {MODEL_ROOT} doesn't exist.", allow_module_level=True)


@fixture(scope="session")
def tagger() -> ITagger:
    _tagger: ITagger = StanzaTagger(
        model=MODEL_ROOT,
        preprocessors=[pretokenize],
        processors="tokenize,lemma,pos",
        tokenize_pretokenized=True,  # model file for tokenize missing
        lang="sv",
        tokenize_no_ssplit=True,
        use_gpu=True,
    )

    return _tagger


def create_sample_corpus(titles: list[str], pages: list[int]) -> pd.DataFrame:
    corpus: pd.DataFrame = load_bolima(DATA_FILENAME)
    pages: pd.DataFrame = corpus[corpus['title'].isin(titles) & corpus['page'].isin(pages)]
    return pages


def load_sample_corpus() -> pd.DataFrame:
    return create_sample_corpus(TEST_DOCUMENTS, [15, 20])


def create_sample_tagged_issues(tagger: ITagger) -> list[TaggedIssue]:
    corpus: pd.DataFrame = load_sample_corpus()
    data: list[TaggedIssue] = []
    for title in TEST_DOCUMENTS:
        issue_pages: pd.DataFrame = corpus[corpus['title'] == title]
        tagged_issue: TaggedIssue = tag_issue(tagger=tagger, title=title, issue_pages=issue_pages, normalize_chars=True)
        data.append(tagged_issue)
    return data


@pytest.mark.skip(reason="Infrastructure test")
def test_store_bolima_sample_corpus():
    corpus: list[pd.DataFrame] = create_sample_corpus(TEST_DOCUMENTS, [15, 20])
    corpus.to_csv(SAMPLE_CORPUS_FILENAME, sep='\t')


@pytest.mark.skip(reason="Infrastructure test")
def test_store_tagged_issue_sample_data(tagger: ITagger):
    tagged_issues: list[TaggedIssue] = create_sample_tagged_issues(tagger)
    for tagged_issue in tagged_issues:
        tagged_issue.tagged_frame.to_csv(f'tests/test_data/{tagged_issue.title}.csv', '\t')
        tagged_issue.document_index.to_csv(f'tests/test_data/{tagged_issue.title}_document_index.csv', '\t')


def test_create_tagger(tagger: ITagger):
    texts: list[str] = [
        "hans lätt och som gol om glädje poetiska",
        "alla förkunnad den i stormen: upprepandet",
        "sin krigsaktivismen \"syskonen\" som varmed för",
        "hemlighet stått handlingsmänniskornas och",
    ]
    data = tagger.tag(texts)
    assert data


def test_tag_bolima(tagger: ITagger):
    corpus: pd.DataFrame = load_sample_corpus()
    for title in TEST_DOCUMENTS:
        issue_pages: pd.DataFrame = corpus[corpus['title'] == title]
        tagged_issue: TaggedIssue = tag_issue(tagger=tagger, title=title, issue_pages=issue_pages, normalize_chars=True)
        assert tagged_issue.title == title
        assert set(tagged_issue.document_index['title']) == {title}
        assert set(tagged_issue.tagged_frame.columns) == {"token", "lemma", "pos", "document_id"}


def test_tagged_issue():
    assert set(TaggedIssue.find('tests/test_data')) == set(TEST_DOCUMENTS)
    assert {x.title for x in TaggedIssue.load_all('tests/test_data')} == set(TEST_DOCUMENTS)
    tagged_issue: TaggedIssue = TaggedIssue.load('tests/test_data', 'BLM-1943:1')
    assert tagged_issue.title == 'BLM-1943:1'
    assert set(tagged_issue.document_index.title) == {'BLM-1943:1'}
    assert tagged_issue.title == 'BLM-1943:1'
    assert set(tagged_issue.document_index.columns) == {
        'index',
        'n_tokens',
        'document_name',
        'title',
        'document_id',
        'n_words',
        'issue_name',
        'page',
        'year',
    }
    assert set(tagged_issue.document_index.document_name) == {'BLM-1943:1_15', 'BLM-1943:1_20'}


def test_issue_reader():
    reader: Iterable[tuple[str, pd.DataFrame]] = issue_reader(source=load_sample_corpus())

    data = list(reader)

    assert len(data) == 2
    assert [x[0] for x in data] == sorted(TEST_DOCUMENTS)


# def test_stanza_download():
#     stanza.download(
#         lang="en",
#         model_dir="/data/stanza_resources",
#         # package='tokenize,mwt'
#         # lang: str = 'en',
#         # model_dir: str = DEFAULT_MODEL_DIR,
#         # package: str = 'default',
#         # processors: Any = {},
#         # logging_level: Any | None = None,
#         # verbose: Any | None = None,
#         # resources_url: str = DEFAULT_RESOURCES_URL,
#         # resources_branch: Any | None = None,
#         # resources_version: str = DEFAULT_RESOURCES_VERSION,
#         # model_url: str = DEFAULT_MODEL_URL
#     )
