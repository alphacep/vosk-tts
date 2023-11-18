# Vosk TTS

Simple TTS based on VITS with some old ideas

## Usage

### Command line

```
pip3 install vosk-tts

vosk-tts -n vosk-model-tts-ru-0.4-natasha --input "Привет мир!" --output out.wav
```

### API

```
from vosk_tts import Model, Synth
model = Model(model_name="vosk-model-tts-ru-0.4-natasha")
synth = Synth(model)

synth.synth("Привет мир!", "out.wav")
```

## Voices

For now we support several Russian voices

| Voice                                                                                                  | Example    |
|--------------------------------------------------------------------------------------------------------|------------|
|[vosk-model-tts-ru-0.4-natasha](https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.4-natasha.zip)  | https://alphacephei.com/tts/vosk-model-tts-0.4-natasha.wav |
|[vosk-model-tts-ru-0.4-irina](https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.4-irina.zip)      | https://alphacephei.com/tts/vosk-model-tts-0.4-irina.wav |
