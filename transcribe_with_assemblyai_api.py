#! /usr/bin/env python3

import argparse
import os
import time
from pathlib import Path
from pprint import pp

from loguru import logger

import assemblyai as aai
aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]


def rm_zh_spaces(text: str) -> str:
    import re
    return re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)


def transcribe(audio_path: Path, language: str = None, format: str = "srt"):
    t0 = time.time()
    audio_path = Path(audio_path)
    logger.info(f"transcribing {audio_path} with language={language}")

    if language:
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.universal,
            language_code=language,
            punctuate=True,
            format_text=True,
        )
    else:
        options = aai.LanguageDetectionOptions(expected_languages=["zh", "en"])
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.universal,
            language_detection=True,
            language_detection_options=options,
            punctuate=True,
            format_text=True,
        )
    transcriber = aai.Transcriber(config=config)
    transcript_obj: aai.transcriber.Transcript = transcriber.transcribe(str(audio_path))

    if transcript_obj.status == "error":
        raise RuntimeError(f"AssemblyAI transcribe failed for {audio_path}: {transcript_obj.error}")

    # real language
    # language = transcript_obj.language_code
    pp(transcript_obj)

    if not language:
        language = transcript_obj.json_response.get("language_code")


    text = transcript_obj.text or ""
    logger.info(f"assemblyai transcribe text with {len(text)}c: \n{text[:800]}\n")

    if format.lower().strip() == "srt":
        srt_text = transcript_obj.export_subtitles_srt(
            chars_per_caption=200,
        )
        if language == "zh":
            # chinse will have extra spaces like:
            # 韋昌輝 殺 死 楊秀卿 之 後, 想要 取代 楊秀卿 的 位置 他
            srt_text = rm_zh_spaces(srt_text)

        print(srt_text)

        srt_path = audio_path.with_suffix(f".{language}.srt" if language else ".srt")
        srt_path.write_text(srt_text)
        t1 = time.time()
        t = t1 - t0
        logger.info(f"SRT/SubRip saved to {srt_path} in {t:.1f}s")
    else:
        raise ValueError(f"format not supported yet: {format}")


def cli():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_path", type=str, help="path to audio file")
    parser.add_argument("-l", "--language", help="language, e.g. en, zh, None for auto-detect")
    parser.add_argument("-f", "--format", default="srt", help="output format")
    return parser.parse_args()


def main():
    args = cli()
    transcribe(args.audio_path, language=args.language, format=args.format)


if __name__ == "__main__":
    main()