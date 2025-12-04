# Docker Subtitles

A Docker Env to generate subtitles with AI models.

Supported models:
- faster whisper (local model, GPU required)
- openai whisper (local model, GPU required)
- openai transcription api (api key required)

Supported subtitle formats:
- srt/SubRip
- vtt/WebVTT
- txt

To be continued.


## Usage

On your Linux machine, in repo root:

```
mkdir data/  # put your video/audio files here.
cp path/to/audio.m4a data/audio.m4a

make build
make shell
```

Now you should be in the container, run any:
```
./openai_whisper_cli.py [-f srt|vtt|txt] [-l zh|en] data/audio.m4a
./faster_whisper_cli.py [-f srt|vtt|txt] data/audio.m4a
```

Example output file: `data/audio.zh.srt`