from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import glob
import os
import argparse
import torch
import torchaudio
from timeit import default_timer as timer

device = 'cuda'

import torch
import bigvgan
from meldataset import get_mel_spectrogram

model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
device = 'cuda'
model.remove_weight_norm()
model = model.eval().to(device)

def inference(a):
    filelist = os.listdir(a.input_wavs_dir)
    os.makedirs(a.output_dir, exist_ok=True)

    tot_samples = 0
    start = timer()
    with torch.no_grad():
        for i, filname in enumerate(filelist):

            y, sr = torchaudio.load(os.path.join(a.input_wavs_dir, filname))
            tot_samples += y.size(1)
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

            mel = get_mel_spectrogram(y, model.h).to(device)
            with torch.inference_mode():
                wav_gen = model(mel)
                wav_gen_float = wav_gen.squeeze(0).cpu()

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            torchaudio.save(output_file, wav_gen_float, 24000, encoding="PCM_S", bits_per_sample=16)
            print(output_file)

    end = timer()
    print (f"Processed {tot_samples / 22050.:.3f} seconds of speech in {end-start} seconds, xRT {(end-start) / (tot_samples / 22050.):3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    a = parser.parse_args()
    inference(a)

if __name__ == '__main__':
    main()

