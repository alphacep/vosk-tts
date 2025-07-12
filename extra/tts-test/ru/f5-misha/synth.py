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

model_cfg = OmegaConf.load("f5-misha-model/F5TTS_v1_Base.yaml")
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
ode_method = 'euler'
device = 'cuda'

vocoder = load_vocoder(mel_spec_type, False, None, device, hf_cache_dir=None)
ema_model = load_model(model_cls, model_arc, "f5-misha-model/model_ruaccent.safetensors", mel_spec_type, "f5-misha-model/vocab.txt", 'euler', True, device)

from ruaccent import RUAccent
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)

spkmap = {}
for i, line in enumerate(open("eval-speakers-text/metadata-phones-ids.csv.vc")):
    items = line.strip().split("|")
    spkmap[items[1]] = (items[0], items[2])

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = accentizer.process_all(items[2].replace("+", ""))

        reffn, reftext = spkmap[items[1]]
        reftext = accentizer.process_all(reftext.replace("+", ""))
        reffn = "eval-speakers-text/wav/" + reffn

        ref_file, ref_text = preprocess_ref_audio_text(reffn, reftext)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            text,
            ema_model,
            vocoder,
            mel_spec_type,
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=2,
            sway_sampling_coef=-1,
            speed=1.0,
            fix_duration=None,
            device='cuda'
        )
        sf.write((out_dir + "/" + uid + ".wav"), wav, sr)

        total_len += os.path.getsize(out_dir + "/" + uid + ".wav") / 2

    end = timer()

    audio_duration_sec = float(total_len) / target_sample_rate
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
