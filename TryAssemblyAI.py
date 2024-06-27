# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai

aai.settings.api_key = "4de1d297d4a24681bc970d28c4a9ced2"
config = aai.TranscriptionConfig(language_code="it")
transcriber = aai.Transcriber(config=config)

transcript = transcriber.transcribe("test.wav")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)