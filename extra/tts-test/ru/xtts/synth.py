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
from TTS.api import TTS

tts = TTS("xtts_v2").to('cuda')

spkmap = {}
for i, line in enumerate(open("eval-speakers/spk.list")):
    spkmap[i] = "eval-speakers/" + line.strip()

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = items[2].replace("+", "")

        tts.tts_to_file(text=text, speaker_wav=spkmap[int(items[1])], language="ru", file_path=out_dir + "/" + uid + ".wav", repetition_penalty=5.0, temperature=0.75)
        total_len += os.path.getsize(out_dir + "/" + uid + ".wav") / 2

    end = timer()

    audio_duration_sec = float(total_len) / 24000
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
