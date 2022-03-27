import uuid
from os.path import isfile, join

import pytest

from pybolima.dispatch import DispatchOptions, TaggedFramePerGroupDispatcher
from pybolima.interface import TaggedIssue
from pybolima.transform import normalize_characters
from pybolima.utility import replace_extension
from pybolima.workflow import tag_bolima

# pylint: disable=redefined-outer-name


@pytest.mark.parametrize(
    'compress_type,skip_text,skip_stopwords',
    [
        ('feather', True, True),
        ('feather', False, False),
        ('csv', True, True),
        ('csv', False, False),
    ],
)
def test_dispatch_bolima(tagged_issues: list[TaggedIssue], compress_type: str, skip_text: bool, skip_stopwords: bool):
    tagged_issue: TaggedIssue = tagged_issues[0]
    target_folder: str = f'tests/output/{str(uuid.uuid4())[:8]}'
    opts: DispatchOptions = DispatchOptions(
        compress_type=compress_type,
        to_lower=True,
        skip_text=skip_text,
        skip_stopwords=skip_stopwords,
        skip_puncts=True,
        skip_lemma=False,
    )
    with TaggedFramePerGroupDispatcher(target=target_folder, opts=opts) as dispatcher:
        dispatcher.dispatch(tagged_issue=tagged_issue)

    assert isfile(join(target_folder, replace_extension(tagged_issue.filename, compress_type)))
    assert isfile(join(target_folder, f'document_index.{compress_type}'))


def test_normalize_characters():

    text = "räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗RÄKSMÖRGÅS"
    normalized_text = normalize_characters(text)
    assert normalized_text == 'räksmörgås----------++//~~~~~~~\'\'\'\'\'\'\'\'\'""""`′′′′′′RÄKSMÖRGÅS'

    text = "räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗RÄKSMÖRGÅS"
    normalized_text = normalize_characters(text, groups="double_quotes,tildes")
    assert normalized_text == 'räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕~~~~~~~’՚Ꞌꞌ＇‘’‚‛""""´″‴‵‶‷⁗RÄKSMÖRGÅS'


def test_workflow():

    args: dict = {
        'numeric_frame': True,
        'source_filename': './data/westac/blm/blm.parquet',
        'target_folder': './tests/output/apa',
        'force': True,
        'compress_type': 'feather',
        'to_lower': True,
        'skip_text': True,
        'skip_stopwords': False,
        'skip_puncts': True,
        'skip_lemma': False,
        'model_root': '/data/sparv/models/stanza',
    }

    tag_bolima(**args)
