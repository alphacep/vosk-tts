#!/usr/bin/env python3

import sys
import os
import torch
import json
import math
import time
from timeit import default_timer as timer
import re
import numpy as np
from scipy.io.wavfile import write

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel

#device = 'cuda'
device = 'cpu'

spectrogram_generator = FastPitchModel.restore_from("tts_ru_ipa_fastpitch_ruslan/tts_ru_ipa_fastpitch_ruslan.nemo").eval().to(device)
vocoder = HifiGanModel.restore_from("tts_ru_hifigan_ruslan/tts_ru_hifigan_ruslan.nemo").eval().to(device)


def clean_russian_g2p_trascription(text: str) -> str:
    result = text
    result = result.replace("", " ").replace("+", "").replace("~", "")
    result = result.replace("ʑ", "ɕ:").replace("ɣ", "x")
    result = result.replace(":", "ː").replace("'", "`")
    result = "".join(result.split())
    result = result.replace("_", " ")
    return result

heteronyms = set()
with open("ru_g2p_ipa_bert_large/heteronyms.txt", "r", encoding="utf-8") as f:
    for line in f:
        inp = line.strip()
        heteronyms.add(inp)

g2p_vocab = {}
# then override known transcriptions using vocabulary
with open("ru_g2p_ipa_bert_large/g2p_correct_vocab.txt", "r", encoding="utf-8") as f:
    for line in f:
        # Example input: ледок \t lʲɪd`ok
        inp, transcription = line.strip().split("\t")
        g2p_vocab[inp] = transcription

def phonetise(line):
    text = line.strip().lower()
    phonemized_text = ""
    m = re.search(r"[\w\-]+", text)
    while m is not None:
        begin = m.start()
        end = m.end()
        phonemized_text += text[0:begin]
        w = text[begin:end]
        if w in heteronyms:
            phonemized_text += w
        elif w in g2p_vocab:
            phonemized_text += clean_russian_g2p_trascription(g2p_vocab[w])
        else:  # shouldn't go here as all words are expected to pass through g2p
            phonemized_text += w

        if end >= len(text):
            break
        text = text[end:]
        end = 0
        m = re.search(r"[\w\-]+", text)
    if end < len(text):
        phonemized_text += text[end:]
        
    return phonemized_text


def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = items[2].replace("+", "")
        text = phonetise(text)
        print (text)

        parsed = spectrogram_generator.parse(text)
        spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed)
        output = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        audio = np.int16(output.detach().cpu().numpy()[0] * 32768)
        total_len = total_len + len(audio)
        write(out_dir + "/" + uid + ".wav", 22050, audio)

    end = timer()

    audio_duration_sec = float(total_len) / 22050
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
