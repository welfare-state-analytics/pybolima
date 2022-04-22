from __future__ import annotations

import os
import shutil
from os.path import isdir, isfile

import pandas as pd

from pybolima.dispatch import DispatchOptions, IdTaggedFramePerGroupDispatcher, TaggedFramePerGroupDispatcher
from pybolima.stanza import ITagger, StanzaTagger
from pybolima.tagger import tag_issues
from pybolima.utility import pretokenize

DEFAULT_MODEL_ROOT: str = "/data/sparv/models/stanza"

# pylint: disable=too-many-arguments


class WorkFlowError(Exception):
    ...


def tag_bolima(
    numeric_frame: bool,
    source_filename: str | pd.DataFrame,
    target_folder: str,
    force: bool = False,
    compress_type: str = 'feather',
    to_lower: bool = True,
    skip_text: bool = True,
    skip_stopwords: bool = False,
    skip_puncts: bool = True,
    skip_lemma: bool = False,
    model_root: str = DEFAULT_MODEL_ROOT,
):
    print(locals())
    if not isfile(source_filename):
        raise FileNotFoundError(source_filename)

    if isdir(target_folder):
        if force:
            shutil.rmtree(target_folder, ignore_errors=True)
        else:
            raise WorkFlowError("target folder exists")

    os.makedirs(target_folder)

    dispatch_cls = IdTaggedFramePerGroupDispatcher if numeric_frame else TaggedFramePerGroupDispatcher
    tagger: ITagger = StanzaTagger(
        model=model_root or DEFAULT_MODEL_ROOT,
        preprocessors=[pretokenize],
        processors="tokenize,lemma,pos",
        tokenize_pretokenized=True,
        lang="sv",
        tokenize_no_ssplit=True,
        use_gpu=True,
    )
    opts: DispatchOptions = DispatchOptions(
        compress_type=compress_type,
        to_lower=to_lower,
        skip_text=skip_text,
        skip_stopwords=skip_stopwords,
        skip_puncts=skip_puncts,
        skip_lemma=skip_lemma,
    )

    tag_issues(tagger, source=source_filename, target=target_folder, dispatch_cls=dispatch_cls, dispatch_opts=opts)
