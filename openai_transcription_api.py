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


HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

LEMONFOX_AI_BASE_URL = "https://api.lemonfox.ai/v1"

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "whisper-1")

# tiny|base|small|turbo|medium|large|turbo
# ref: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
model_size = "turbo"  # optimized large-v3

logger.info(f"loading model {model_size}")
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
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
        if OPENAI_BASE_URL == LEMONFOX_AI_BASE_URL:
            logger.info("convert lemonfox result from json to python str")
            result = json.loads(result)

        print(result)

    if format.lower().strip() == "srt":
        suffix = f".{language}.srt" if language else ".srt"
        srt_path = audio_path.with_suffix(suffix)
        srt_path.write_text(result)
        t = time.time() - t0
        logger.info(f"SRT/SubRip saved to {srt_path} in {t:.1f}s")
    else:
        raise ValueError(f"format not supported yet: {format}")


def cli():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_path", type=str, help="path to audio file")
    parser.add_argument("-l", "--language", help="language, e.g. en, zh")
    parser.add_argument("-f", "--format", default="srt", help="output format")
    return parser.parse_args()


def main():
    args = cli()
    language = args.language
    while not language:
        language = input("Enter language (e.g.: en, zh): ")
    transcribe(args.audio_path, language=language, format=args.format)


if __name__ == "__main__":
    main()