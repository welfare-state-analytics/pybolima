from __future__ import annotations

import bz2
import gzip
import lzma
import os
from os.path import basename, splitext
from typing import Literal

import numpy as np
import pandas as pd

from pybolima.foss.sparv_tokenize import default_tokenize


def pretokenize(text: str) -> str:
    """Tokenize `text`, then join resulting tokens."""
    return ' '.join(default_tokenize(text))


def strip_path_and_extension(filename: str | list[str]) -> str | list[str]:
    """Remove path and extension from filename(s). Return list."""
    if isinstance(filename, str):
        return splitext(basename(filename))[0]
    return [splitext(basename(x))[0] for x in filename]


def strip_extensions(filename: str | list[str]) -> str | list[str]:
    if isinstance(filename, str):
        return splitext(filename)[0]
    return [splitext(x)[0] for x in filename]


def replace_extension(filename: str, extension: str) -> str:
    if filename.endswith(extension):
        return filename
    base, _ = os.path.splitext(filename)
    return f"{base}{'' if extension.startswith('.') else '.'}{extension}"


def store_str(filename: str, text: str, compress_type: Literal['csv', 'gzip', 'bz2', 'lzma']) -> None:
    """Stores a textfile on disk - optionally compressed"""
    modules = {'gzip': (gzip, 'gz'), 'bz2': (bz2, 'bz2'), 'lzma': (lzma, 'xz')}

    if compress_type in modules:
        module, extension = modules[str(compress_type)]
        with module.open(f"{filename}.{extension}", 'wb') as fp:
            fp.write(text.encode('utf-8'))

    elif compress_type == 'csv':
        with open(filename, 'w', encoding='utf-8') as fp:
            fp.write(text)
    else:
        raise ValueError(f"unknown mode {compress_type}")


def trim_series_type(series: pd.Series) -> pd.Series:
    max_value: int = series.max()
    for np_type in [np.int16, np.int32]:
        if max_value < np.iinfo(np_type).max:
            return series.astype(np_type)
    return series
