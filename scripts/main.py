from __future__ import annotations
import sys

import click

from pybolima import workflow


@click.command()
@click.argument('source_filename', type=click.STRING)
@click.argument('target_folder', type=click.STRING)
@click.option('--codify', type=click.BOOL, is_flag=True, help='Codified frame', default=True)
@click.option('--force', type=click.BOOL, is_flag=True, help='Force overwrite', default=False)
@click.option('--compress-type', type=click.STRING, help='Storage format', default='feather')
@click.option('--to-lower', type=click.BOOL, is_flag=True, help='Lowercase tokens', default=True)
@click.option('--skip-text', type=click.BOOL, is_flag=True, help='Skip text column', default=True)
@click.option('--skip-stopwords', type=click.BOOL, is_flag=True, help='Skip stopwords', default=False)
@click.option('--skip-puncts', type=click.BOOL, is_flag=True, help='Skip punctuations', default=True)
@click.option('--skip-lemma', type=click.BOOL, is_flag=True, help='Skip lemma', default=False)
@click.option('--model-root', type=click.STRING, default=workflow.DEFAULT_MODEL_ROOT)
def main(
    source_filename: str,
    target_folder: str,
    codify: bool = True,
    force: bool = False,
    compress_type: str = 'feather',
    to_lower: bool = True,
    skip_text: bool = True,
    skip_stopwords: bool = False,
    skip_puncts: bool = True,
    skip_lemma: bool = False,
    model_root: str = None,
) -> None:
    try:

        workflow.tag_bolima(
            source_filename=source_filename,
            target_folder=target_folder,
            numeric_frame=codify,
            force=force,
            compress_type=compress_type,
            to_lower=to_lower,
            skip_text=skip_text,
            skip_stopwords=skip_stopwords,
            skip_puncts=skip_puncts,
            skip_lemma=skip_lemma,
            model_root=model_root,
        )

    except Exception as ex:
        click.echo(ex)
        sys.exit(1)


# type: ignore

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
