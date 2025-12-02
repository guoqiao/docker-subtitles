# Docker Subtitles

A Docker Env to generate subtitles with AI models.

Supported models:
- openai whisper local
- openai whisper api key
- faster-whisper

Supported subtitle formats:
- srt/SubRip

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
./faster_whisper_cli.py data/audio.m4a
./openai_whisper_cli.py data/audio.m4a
```

Example output file: `data/audio.zh.srt`