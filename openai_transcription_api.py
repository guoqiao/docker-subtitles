#! /usr/bin/env python3
"""Generate subtitles with openai whisper api remotely."""

import argparse
import os
import json
import time
from pathlib import Path
from pprint import pp

from loguru import logger
from openai import OpenAI
from openai.types.audio.transcription import Transcription

# https://github.com/openai/openai-python/blob/main/src/openai/types/audio/translation_verbose.py
from openai.types.audio.transcription_verbose import TranscriptionVerbose


HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "whisper-1")

OPENAI_API_BACKEND_LEMONFOX = "lemonfox"
OPENAI_API_BACKEND = os.getenv("OPENAI_API_BACKEND", OPENAI_API_BACKEND_LEMONFOX)

if OPENAI_API_BACKEND == OPENAI_API_BACKEND_LEMONFOX:
    logger.info("using lemonfox api")
    client = OpenAI(
        base_url="https://api.lemonfox.ai/v1",
        api_key=os.environ["LEMONFOX_AI_API_KEY"],
    )
else:
    logger.info("using openai api")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )


def transcribe(audio_path: Path, language: str = None, format: str = "srt"):
    t0 = time.time()
    audio_path = Path(audio_path)
    logger.info(f"transcribing {audio_path} with {OPENAI_MODEL_NAME}")
    with audio_path.open("rb") as audio_file:
        # https://platform.openai.com/docs/api-reference/audio/createTranscription
        result = client.audio.transcriptions.create(
            model=OPENAI_MODEL_NAME,
            file=audio_file,
            # https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
            language=language,
            # verbose json can return language, but with more tokens
            # https://platform.openai.com/docs/api-reference/audio/verbose-json-object
            response_format=format,
        )

    pp(result)

    fmt = format.lower().strip()
    if OPENAI_API_BACKEND == OPENAI_API_BACKEND_LEMONFOX:
        if fmt in ["vtt", "srt", "text"]:
            logger.info("convert lemonfox result from json to python str")
            result = json.loads(result)

    # when request json format, lemonfox returns a Transcription object
    if isinstance(result, Transcription):
        fmt = "text"  # json
        result = result.text
    elif isinstance(result, TranscriptionVerbose):
        fmt = "text"  # verbose_json
        result = result.text

    ext = FORMAT_EXT.get(fmt, fmt)
    suffix = f".{language}.{ext}" if language else f".{ext}"
    out_path = audio_path.with_suffix(suffix)

    out_path.write_text(result)
    t = time.time() - t0
    logger.info(f"{fmt} saved to {out_path} in {t:.1f}s")


# supported response formats and their file extensions
FORMAT_EXT = {
    "vtt": "vtt",
    "srt": "srt",
    "text": "txt",
    "json": "json",
    "verbose_json": "json",
}


def cli():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_path", type=str, help="path to audio file")
    parser.add_argument("-l", "--language", help="language, e.g. en, zh")
    parser.add_argument("-f", "--format", choices=FORMAT_EXT, default="vtt", help="api response format")
    return parser.parse_args()


def main():
    args = cli()
    language = args.language
    while not language:
        language = input("Enter language (e.g.: en, zh): ")
    transcribe(args.audio_path, language=language, format=args.format)


if __name__ == "__main__":
    main()