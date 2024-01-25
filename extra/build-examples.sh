#!/bin/bash

text='У Лукоморья дуб зелёный. Златая цепь на дубе том. И днём и ночью кот учёный всё ходит по цеп+и круг+ом. Идёт направо, песнь заводит. Налево, сказку говорит.'

for i in {0..4}; do
    vosk-tts -m ../vosk-model-tts-ru-0.6-multi -s $i -i "$text" -o vosk-model-tts-0.6-multi-speaker-$i.wav
done
