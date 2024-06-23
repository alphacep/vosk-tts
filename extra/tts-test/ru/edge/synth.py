#!/usr/bin/env python3

import sys
import os
from timeit import default_timer as timer
import edge_tts

def main_synth(out_dir=None, voice=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        text = items[2].replace("+", "")
        fn = out_dir + "/" + uid

        communicate = edge_tts.Communicate(text, voice)
        communicate.save_sync(f"{fn}.mp3")
        os.system(f"ffmpeg -y -nostdin -i {fn}.mp3 {fn}.wav")
        total_len = total_len + os.path.getsize(fn + ".wav") / 2

    end = timer()

    audio_duration_sec = float(total_len) / 24000
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out_dmitry', voice = "ru-RU-DmitryNeural")
main_synth(out_dir = 'out_svetlana', voice = "ru-RU-SvetlanaNeural")
