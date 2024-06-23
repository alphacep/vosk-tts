#!/usr/bin/env python3

import sys
import os
from timeit import default_timer as timer


from speechkit import configure_credentials, creds
from speechkit import model_repository

configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key='<here>'
    )
)

model = model_repository.synthesis_model()

def main_synth(out_dir=None, voice=None):

    os.makedirs(out_dir, exist_ok=True)
    model.voice = voice

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = items[2].replace("+", "")
        fn = out_dir + "/" + uid + ".wav"

        result = model.synthesize(text, raw_format=False)
        result.export(fn, format='wav')
        total_len = total_len + os.path.getsize(fn) / 2

    end = timer()

    audio_duration_sec = float(total_len) / 22050
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
#main_synth(out_dir = 'out_alexander', voice = "alexander")
#main_synth(out_dir = 'out_marina', voice = "marina")
