# Vosk TTS

Simple TTS based on VITS with some old ideas

## Usage

### Command line

```
pip3 install vosk-tts

vosk-tts -n vosk-model-tts-ru-0.2-natasha --input "Привет мир!" --output ~/out.wav
```

### API

```
from vosk_tts import Model, Synth
model = Model(model_name="vosk-model-tts-ru-0.2-natasha")
synth = Synth(model)

synth.synth("Привет мир!", "test.wav")
```

## Voices

For now we support several Russian voices

| Voice                               | Sample             |
|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
|[vosk-model-tts-ru-0.2-natasha](https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.2-natasha.zip)  | https://github.com/alphacep/vosk-tts/raw/master/samples/vosk-model-tts-ru-0.2-natasha.wav |
|[vosk-model-tts-ru-0.1-irina](https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.1-irina.zip)      | https://github.com/alphacep/vosk-tts/raw/master/samples/vosk-model-tts-ru-0.1-irina.wav   |
|[vosk-model-tts-ru-0.1-tamara](https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.1-tamara.zip)    | https://github.com/alphacep/vosk-tts/raw/master/samples/vosk-model-tts-ru-0.1-tamara.wav  |
