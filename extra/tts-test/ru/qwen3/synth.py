#!/usr/bin/env python3
import os
import sys
from timeit import default_timer as timer

import torchaudio

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "/home/shmyrev/kaldi/egs/ac/ru-tts-test/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base",
    dtype=torch.bfloat16,
    device_map="cuda",
)

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
        text = items[1].replace("+", "")

        reffn, reftext = spkmap[spk]
        reftext = reftext.replace("+", "")
        reffn = "eval-speakers-text/wav/" + reffn

        out_wav, out_sr = model.generate_voice_clone(
            text=text,
            language="Russian",
            ref_audio=reffn,
            ref_text=reftext)
        torchaudio.save(out_dir + "/" + uid + ".wav", torch.from_numpy(out_wav[0]).unsqueeze(0), out_sr)

        total_len += os.path.getsize(out_dir + "/" + uid + ".wav") / 2

    end = timer()

    audio_duration_sec = float(total_len) / 24000
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
