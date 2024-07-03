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

def main_synth(speaker=None, out_dir=None, model=None, device=None):

    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(8)
    device = torch.device(device)

    model = torch.package.PackageImporter(f"models/{model}.pt").load_pickle("tts_models", "model")
    model.to(device)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        audio = model.apply_tts(text=items[2],
                                speaker=speaker,
                                sample_rate=48000)
        audio16 = np.int16(audio * 32767)
        total_len = total_len + len(audio16)
        write(out_dir + "/" + uid + ".wav", 48000, audio16)
    end = timer()

    audio_duration_sec = float(total_len) / 48000
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
#main_synth(speaker = 'baya', out_dir = 'out_baya_v4', model = 'v4_ru', device='cpu')
#main_synth(speaker = 'baya', out_dir = 'out_baya_v3', model = 'v3_1_ru', device='cpu')
#main_synth(speaker = 'baya', out_dir = 'out_baya_v4', model = 'v4_ru', device='cuda')
#main_synth(speaker = 'baya', out_dir = 'out_baya_v3', model = 'v3_1_ru', device='cuda')
main_synth(speaker = 'aidar', out_dir = 'out_aidar_v4', model = 'v4_ru', device='cuda')
main_synth(speaker = 'aidar', out_dir = 'out_aidar_v3', model = 'v3_1_ru', device='cuda')
