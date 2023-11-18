#!/bin/bash

text='У Лукоморья дуб зелёный. Златая цепь на дубе том. И днём и ночью кот учёный всё ходит по цеп+и круг+ом. Идёт направо, песнь заводит. Налево, сказку говорит.'

vosk-tts -n vosk-model-tts-ru-0.4-natasha -i "$text" -o vosk-model-tts-0.4-natasha.wav
vosk-tts -n vosk-model-tts-ru-0.4-irina -i "$text" -o vosk-model-tts-0.4-irina.wav
