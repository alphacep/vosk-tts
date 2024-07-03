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


from ruaccent import RUAccent
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo', use_dictionary=True)
from transformers import VitsModel, AutoTokenizer, set_seed


device = 'cuda'
#device = 'cpu'

set_seed(555)

# load model
model_name = "utrobinmv/tts_ru_free_hf_vits_low_multispeaker"

model = VitsModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

def main_synth(out_dir=None, speaker=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = accentizer.process_all(items[2])

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
          output = model(**inputs.to(device), speaker_id=speaker).waveform
          audio = np.int16(output.detach().cpu().numpy()[0] * 32768)
          total_len = total_len + len(audio)
          write(out_dir + "/" + uid + ".wav", model.config.sampling_rate, audio)

    end = timer()

    audio_duration_sec = float(total_len) / model.config.sampling_rate
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out0', speaker = 0)
main_synth(out_dir = 'out1', speaker = 1)
