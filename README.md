# Docker Transcribers

A Docker Env to transcribe video/audio with AI model/API and generate subtitles.

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
./transcribe_with_faster_whisper.py [-f srt|vtt|txt] data/audio.m4a
./transcribe_with_openai_whisper.py [-f srt|vtt|txt] data/audio.m4a
./transcribe_with_openai_api.py [-f srt|vtt|txt] data/audio.m4a
```

Example output file: `data/audio.zh.srt`