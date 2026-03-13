#!/usr/bin/env python3

import sys
import os
import json
import math
import time
from timeit import default_timer as timer

import torch
import numpy as np
import soundfile as sf

from omegaconf import OmegaConf
from hydra.utils import get_class

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
)
from f5_tts.model.utils import seed_everything

seed_everything(1234)

device = 'cuda'

vocoder = load_vocoder()

from f5_tts.model import DiT

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
ema_model = load_model(DiT, MODEL_CFG, "f5-denis-podcaster/espeech_tts_podcaster.pt" , vocab_file="f5-denis-podcaster/vocab.txt").float()

from ruaccent import RUAccent
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)

spkmap = {}
for i, line in enumerate(open("eval-speakers-text/metadata-phones-ids.csv.test-ref")):
    items = line.strip().split("|")
    spk = items[0].split("/")[0]
    spkmap[spk] = (items[0].replace("/", "_"), items[1])

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        spk = items[0].split("/")[2]
        text = accentizer.process_all(items[1].replace("+", ""))

        reffn, reftext = spkmap[spk]
        reftext = accentizer.process_all(reftext.replace("+", ""))
        reffn = "eval-speakers-text/wav/" + reffn

        ref_file, ref_text = preprocess_ref_audio_text(reffn, reftext)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            text,
            model,
            vocoder,
            nfe_step=16,
            device='cuda'
        )
        sf.write((out_dir + "/" + uid + ".wav"), wav, sr)

        total_len += os.path.getsize(out_dir + "/" + uid + ".wav") / 2

    end = timer()

    audio_duration_sec = float(total_len) / 24000
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
