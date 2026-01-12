from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import glob
import os
import argparse
import torch
import torchaudio
from timeit import default_timer as timer

from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio

device = 'cuda'

wavtokenizer = WavTokenizer.from_pretrained0802("WavTokenizer-large-speech-75token/config.yaml", "WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_24k.ckpt")
wavtokenizer = wavtokenizer.to(device)

def inference(a):
    filelist = os.listdir(a.input)
    os.makedirs(a.output, exist_ok=True)

    tot_samples = 0
    start = timer()

    with torch.no_grad():
        for i, filname in enumerate(filelist):

            wav, sr = torchaudio.load(os.path.join(a.input, filname))
            wav = convert_audio(wav, sr, 24000, 1) 
            bandwidth_id = torch.tensor([0]).to(device)
            tot_samples += wav.size(1)
            wav = wav.to(device)

            features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
            audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id).cpu()
            output_file = os.path.join(a.output, os.path.splitext(filname)[0] + '_generated.wav')
            torchaudio.save(output_file, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
            print(output_file)

    end = timer()
    print (f"Processed {tot_samples / 24000.:.3f} seconds of speech in {end-start} seconds, xRT {(end-start) / (tot_samples / 24000.):3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='test_files')
    parser.add_argument('--output', default='generated_files')
    a = parser.parse_args()
    inference(a)

if __name__ == '__main__':
    main()

