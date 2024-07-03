#!/usr/bin/env python3

import sys
import os
import torch
import json
import math
import time
from timeit import default_timer as timer
import numpy as np
from scipy.io.wavfile import write

from TeraTTS import TTS

from ruaccent import RUAccent
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo', use_dictionary=True)

tts = TTS("TeraTTS/natasha-g2p-vits", add_time_to_end=0.5, tokenizer_load_dict=True)

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = accentizer.process_all(items[2])

        audio = tts(text, lenght_scale=1.1)
        total_len = total_len + len(audio)
        write(out_dir + "/" + uid + ".wav", 22050, audio)
    end = timer()

    audio_duration_sec = float(total_len) / 22050
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
