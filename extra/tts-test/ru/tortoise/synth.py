#!/usr/bin/env python3

import sys
import os
import torch
import json
import math
import time
import torchaudio
from timeit import default_timer as timer
import numpy as np
from scipy.io.wavfile import write

from api import TextToSpeech
from utils.audio import load_audio

spkmap = {}
for i, line in enumerate(open("eval-speakers/spk.list")):
    spkmap[i] = "eval-speakers/" + line.strip()

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    tts = TextToSpeech(models_dir='model-tortoise-ru')


    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = items[2].replace("+", "")

        reference_clips = [load_audio(spkmap[int(items[1])], 22050)]
        pcm_audio = tts.tts_with_preset(text, voice_samples=reference_clips, preset='ultra_fast')
        pcm_audio = pcm_audio.squeeze(0).cpu()
        pcm_audio = (pcm_audio * 32768).to(torch.int16)
        print (pcm_audio.size())
        torchaudio.save((out_dir + "/" + uid + ".wav"), pcm_audio, 24000)
        total_len += pcm_audio.size(1)

    end = timer()

    audio_duration_sec = float(total_len) / 24000
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
