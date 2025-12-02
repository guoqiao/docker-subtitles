#! /usr/bin/env python3

import argparse
import time
from pathlib import Path

from loguru import logger
from faster_whisper import WhisperModel


HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

model_size = "large-v3"
compute_type = "float16"  # float16|int8|int8_float16
beam_size = 5

logger.info(f"loading model {model_size} with {compute_type}")
model = WhisperModel(
    model_size,
    device="cuda",  # cuda|cpu
    compute_type=compute_type,
    download_root=str(MODELS_DIR),
)


def sec2srt(seconds):
    # 3661.234 -> 01:01:01,234
    n = int(seconds)
    h, s = divmod(n, 3600)
    m, s = divmod(s, 60)
    ms = int((seconds - n) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def transcribe(audio_path: Path, language: str = None, format: str = "srt"):
    t0 = time.time()
    audio_path = Path(audio_path)
    logger.info(f"transcribing {audio_path} with beam_size={beam_size}, language={language}")
    # https://github.com/SYSTRAN/faster-whisper/blob/ed9a06cd89a93e47838f564998a6c09b655d7f43/faster_whisper/transcribe.py#L254
    segments, info = model.transcribe(str(audio_path), beam_size=beam_size, language=language)

    if not language:
        logger.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        language = info.language

    if format.lower().strip() == "srt":
        captions = []
        for i, segment in enumerate(segments, start=1):
            caption = "\n".join([
                f"{i}",
                f"{sec2srt(segment.start)} --> {sec2srt(segment.end)}",
                segment.text.strip(),
            ])
            print(caption, end="\n\n")
            captions.append(caption)

        srt_path = audio_path.with_suffix(f".{language}.srt")
        srt_text = "\n\n".join(captions)
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