#!/usr/bin/env python3

import sys
import os
import json
import math
import time
from timeit import default_timer as timer

from piper import PiperVoice
import wave

def main_synth(model = None, out_dir=None):

    voice = PiperVoice.load(model, config_path=model + ".json")

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = items[2].replace("+", "")

        with wave.open(out_dir + "/" + uid + ".wav", "wb") as wav_file:
           voice.synthesize(text, wav_file)
        total_len = total_len + os.path.getsize(out_dir + "/" + uid + ".wav") / 2

    end = timer()

    audio_duration_sec = float(total_len) / 22050
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(model = "piper-voices/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx", out_dir = 'out_denis')
main_synth(model = "piper-voices/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx", out_dir = 'out_dmitri')
main_synth(model = "piper-voices/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx", out_dir = 'out_irina')
main_synth(model = "piper-voices/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx", out_dir = 'out_ruslan')
