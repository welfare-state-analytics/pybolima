import pandas as pd

from pybolima.load import issue_reader, load_bolima

from . import DATA_FILENAME, SAMPLE_CORPUS_FILENAME

EXPECTED_COLUMNS: set[str] = {
    'issue_name',
    'text',
    'document_name',
    'page',
    'document_id',
    'year',
    'title',
}  # , 'dark_id'


def test_load():

    data = load_bolima(DATA_FILENAME)
    assert data is not None
    assert len(data) == 46360
    assert EXPECTED_COLUMNS.intersection(data.columns) == EXPECTED_COLUMNS

    data = load_bolima(SAMPLE_CORPUS_FILENAME)
    assert data is not None
    assert EXPECTED_COLUMNS.intersection(data.columns) == EXPECTED_COLUMNS


def test_issue_reader():

    issues = [x for x in issue_reader(SAMPLE_CORPUS_FILENAME)]
    assert issues is not None
    assert len(issues) == 2

    title, data = issues[0]
    assert title == 'BLM-1943:1'
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2

    assert EXPECTED_COLUMNS.intersection(data.columns) == EXPECTED_COLUMNS
