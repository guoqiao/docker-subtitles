# Docker Subtitles

A Docker Env to generate subtitles with AI models.

Supported models:
- faster-whisper

To be continued.


## Usage

On your Linux machine, in repo root:

```
mkdir data/  # put your video/audio files here.
cp path/to/audio.m4a data/audio.m4a

make build
make shell
```

Now you should be in the container, run:
```
./faster_whisper_cli.py data/audio.m4a
```

Example output file: `data/audio.zh.srt`