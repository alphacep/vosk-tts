#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/shmyrev/kaldi/egs/ac/ru-tts-test/fish/fish-speech')
import os
import torch
import json
import math
import time
from timeit import default_timer as timer
import numpy as np
from scipy.io.wavfile import write


from tools.llama.generate import launch_thread_safe_queue
from tools.vqgan.inference import load_model as load_decoder_model
from tools.inference_engine import TTSInferenceEngine
from tools.schema import ServeTTSRequest, ServeReferenceAudio


device='cuda'
precision=torch.float32
model_root='/home/shmyrev/kaldi/egs/ac/ru-tts-test/fish/fish-speech/checkpoints/fish-speech-1.5'

print("Loading Llama model...")
llama_queue = launch_thread_safe_queue(
        checkpoint_path=model_root,
        device=device,
        precision=precision,
        compile=False,
)

print("Loading VQ-GAN model...")
decoder_model = load_decoder_model(
    config_name='firefly_gan_vq',
    checkpoint_path=model_root + '/firefly-gan-vq-fsq-8x1024-21hz-generator.pth',
    device=device,
)

# Create the inference engine
inference_engine = TTSInferenceEngine(
    llama_queue=llama_queue,
    decoder_model=decoder_model,
    compile=False,
    precision=precision,
)



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
        text = items[2].replace("+", "")

        reffn, reftext = spkmap[items[1]]
        reftext = reftext.replace("+", "")
        reffn = "eval-speakers-text/wav/" + reffn

        req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=open(reffn, "rb").read(), text=reftext)],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav")

        for result in inference_engine.inference(req):
            match result.code:
                case "final":
                    audio_array = np.int16(result.audio[1] * 32768)
                    write((out_dir + "/" + uid + ".wav"), 44100, audio_array)
                case "error":
                    print ("!!!! Error " + result.error)
                    break
                case _:
                    pass

        total_len += os.path.getsize(out_dir + "/" + uid + ".wav") / 2

    end = timer()

    audio_duration_sec = float(total_len) / 44100
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
