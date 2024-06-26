#!/usr/bin/env python3

import os

os.environ['SUNO_USE_SMALL_MODELS'] = "1"

import torch
import json
import math
import time
import torchaudio
from timeit import default_timer as timer
import numpy as np
from scipy.io.wavfile import write

from bark import SAMPLE_RATE, generate_audio, preload_models

preload_models()

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = items[2].replace("+", "")

        audio_array = generate_audio(text, history_prompt='v2/ru_speaker_4')
        audio_array = np.int16(audio_array * 32768)
        write((out_dir + "/" + uid + ".wav"), SAMPLE_RATE, audio_array)
        total_len += len(audio_array)

    end = timer()

    audio_duration_sec = float(total_len) / SAMPLE_RATE
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
