"""PoS tagging using Stanford's Stanza library.
NOTE! THIS CODE IS IN PART BASED ON https://github.com/spraakbanken/sparv-pipeline/blob/master/sparv/modules/stanza/stanza.py
"""
import abc
import itertools
import os
from functools import reduce
from typing import Any, Callable, Union

import stanza

jj = os.path.join

TaggedData = dict[str, list[str]]

"""Follow Språkbanken Sparv's naming of model names and config keys."""
STANZA_CONFIGS: dict = {
    "sv": {
        "resources_file": "resources.json",
        "lem_model": jj("lem", "sv_suc_lemmatizer.pt"),
        "pos_model": jj("pos", "full_sv_talbanken_tagger.pt"),
        "pretrain_pos_model": jj("pos", "full_sv_talbanken.pretrain.pt"),
        "dep_model": jj("dep", "sv_talbanken_parser.pt"),
        "pretrain_dep_model": jj("pos", "full_sv_talbanken.pretrain.pt"),
    }
}


class ITagger(abc.ABC):
    def __init__(self, preprocessors: Callable[[str], str] = None):
        self.preprocessors: Callable[[str], str] = preprocessors or []

    def tag(self, text: Union[str, list[str]]) -> list[TaggedData]:
        """Tag text. Return dict if lists."""
        if isinstance(text, str):
            text = [text]

        if not isinstance(text, list):
            return ValueError("invalid type")

        if len(text) == 0:
            return []

        if self.preprocessors:
            text: list[str] = [self.preprocess(d) for d in text]

        tagged_documents = self._tag(text)

        return tagged_documents

    @abc.abstractmethod
    def _tag(self, text: Union[str, list[str]]) -> list[TaggedData]:
        ...

    @abc.abstractmethod
    def _to_dict(self, tagged_document: Any) -> TaggedData:
        return {}

    @staticmethod
    def to_csv(tagged_document: TaggedData, sep='\t') -> str:
        """Converts a TaggedDocument to a TSV string"""

        tokens, lemmas, pos, xpos = (
            tagged_document['token'],
            tagged_document['lemma'],
            tagged_document['pos'],
            tagged_document['xpos'],
        )
        csv_str = '\n'.join(
            itertools.chain(
                [f"token{sep}lemma{sep}pos{sep}xpos"],
                (f"{tokens[i]}{sep}{lemmas[i]}{sep}{pos[i]}{sep}{xpos[i]}" for i in range(0, len(tokens))),
            )
        )
        return csv_str

    def preprocess(self, text: str) -> str:
        """Transform `text` with preprocessors."""
        text: str = reduce(lambda res, f: f(res), self.preprocessors, text)
        return text


class StanzaTagger(ITagger):
    """Stanza PoS tagger wrapper"""

    def __init__(
        self,
        model: str,
        preprocessors: Callable[[str], str],
        lang: str = "sv",
        processors: str = "lemma,pos",
        tokenize_pretokenized: bool = True,
        tokenize_no_ssplit: bool = True,
        use_gpu: bool = True,
    ):
        super().__init__(preprocessors=preprocessors)  ## or [pretokenize])

        """Initialize stanza pipeline

        Args:
            model_root (str): where Språkbanken's Stanza models are stored
            preprocessors (Callable[[str], str]): Text transforms to do prior to tagging.
            lang (str, optional): Language (only 'sv' supported). Defaults to "sv".
            processors (str, optional): Stanza process steps. Defaults to "lemma,pos".
            tokenize_pretokenized (bool, optional): If true, then already tokenized. Defaults to True.
            tokenize_no_ssplit (bool, optional): [description]. Defaults to True.
            use_gpu (bool, optional): If true, use GPU if exists. Defaults to True.
        """
        print(f"stanza: processors={processors} use_gpu={use_gpu}")
        config: dict = STANZA_CONFIGS[lang]
        self.nlp: stanza.Pipeline = stanza.Pipeline(
            lang=lang,
            processors=processors,
            dir=model,
            pos_pretrain_path=jj(model, config["pretrain_pos_model"]),
            pos_model_path=jj(model, config["pos_model"]),
            lemma_model_path=jj(model, config["lem_model"]),
            tokenize_pretokenized=tokenize_pretokenized,
            tokenize_no_ssplit=tokenize_no_ssplit,
            use_gpu=use_gpu,
            verbose=False,
        )

    def _tag(self, text: Union[str, list[str]]) -> list[TaggedData]:
        """Tag text. Return dict if lists."""

        documents: list[stanza.Document] = [stanza.Document([], text=d) for d in text]

        tagged_documents: list[stanza.Document] = self.nlp(documents)

        if isinstance(tagged_documents, stanza.Document):
            tagged_documents = [tagged_documents]

        return [self._to_dict(d) for d in tagged_documents]

    def _to_dict(self, tagged_document: stanza.Document) -> TaggedData:
        """Extract tokens from tagged document. Return dict of list."""

        tokens, lemmas, pos, xpos = [], [], [], []
        for w in tagged_document.iter_words():
            tokens.append(w.text)
            lemmas.append(w.lemma or w.text.lower())
            pos.append(w.upos)
            xpos.append(w.xpos)

        return dict(
            token=tokens,
            lemma=lemmas,
            pos=pos,
            xpos=xpos,
            n_tokens=tagged_document.num_tokens,
            n_words=tagged_document.num_words,
        )
