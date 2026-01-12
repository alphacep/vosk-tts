from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import torch
import torchaudio
from timeit import default_timer as timer

from linacodec.codec import LinaCodec
lina_tokenizer = LinaCodec() ## will download YatharthS/LinaCodec from huggingface

def inference(a):
    filelist = os.listdir(a.input)
    os.makedirs(a.output, exist_ok=True)

    tot_samples = 0
    start = timer()

    with torch.no_grad():
        for i, filname in enumerate(filelist):

            tot_samples += os.path.getsize(a.input + "/" + filname) / 2

            speech_tokens, global_embedding = lina_tokenizer.encode(a.input + "/" + filname)
            audio = lina_tokenizer.decode(speech_tokens, global_embedding).cpu()
            output_file = os.path.join(a.output, os.path.splitext(filname)[0] + '_generated.wav')
            torchaudio.save(output_file, audio, 48000, encoding="PCM_S", bits_per_sample=16)
            print(output_file)

    end = timer()
    print (f"Processed {tot_samples / 22050.:.3f} seconds of speech in {end-start} seconds, xRT {(end-start) / (tot_samples / 22050.):3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='test_files')
    parser.add_argument('--output', default='generated_files')
    a = parser.parse_args()
    inference(a)

if __name__ == '__main__':
    main()

