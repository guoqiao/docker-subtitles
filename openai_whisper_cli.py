#! /usr/bin/env python3
"""Generate subtitles with GPU and openai whisper model locally."""

import argparse
import time
from pathlib import Path
from pprint import pp

from loguru import logger
import whisper


HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

# tiny|base|small|turbo|medium|large|turbo
# ref: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
model_size = "turbo"  # optimized large-v3

logger.info(f"loading model {model_size}")
model = whisper.load_model(
    model_size
)


def sec2srt(seconds):
    # 3661.234 -> 01:01:01,234
    n = int(seconds)
    h, s = divmod(n, 3600)
    m, s = divmod(s, 60)
    ms = int((seconds - n) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def seg2srt(i, segment):
    """Segment to srt caption:

    segment example:
    {'id': 20,
    'seek': 5542,
    'start': 58.9,
    'end': 59.900000000000006,
    'text': '就要杀师大开',
    'tokens': [50539, 3111, 4275, 4422, 222, 29186, 3582, 18937, 50589],
    'temperature': 0.0,
    'avg_logprob': -0.13662971594394782,
    'compression_ratio': 0.9259259259259259,
    'no_speech_prob': 1.473535703178097e-11}
    """
    start = segment.get("start", 0)
    end = segment.get("end", 0)
    text = segment.get("text", "")
    return "\n".join([
        f"{i}",
        f"{sec2srt(start)} --> {sec2srt(end)}",
        text.strip(),
    ])


def transcribe(audio_path: Path, language: str = None, format: str = "srt", verbose: bool = False):
    t0 = time.time()
    audio_path = Path(audio_path)
    logger.info(f"transcribing {audio_path}")
    # ref: https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L38
    result = model.transcribe(
        str(audio_path),
        language=language,
        verbose=verbose,  # will print help info and captions
    )
    # pp(result)

    language = result.get("language")

    # text = result.get("text", "")
    # print(text)

    segments = result.get("segments", [])
    if not segments:
        logger.warning(f"no segments found for {audio_path}")
        return

    if format.lower().strip() == "srt":
        captions = []
        for i, segment in enumerate(segments, start=1):
            caption = seg2srt(i, segment)
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