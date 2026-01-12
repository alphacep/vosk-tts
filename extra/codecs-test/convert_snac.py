from __future__ import absolute_import, division, print_function, unicode_literals


import glob
import os
import argparse
import torch
import torchaudio
from timeit import default_timer as timer

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

device = 'cpu'

from snac import SNAC
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
count_parameters(model)

def inference(a):
    filelist = os.listdir(a.input)
    os.makedirs(a.output, exist_ok=True)

    tot_samples = 0
    start = timer()

    with torch.no_grad():
        for i, filname in enumerate(filelist):

            y, sr = torchaudio.load(os.path.join(a.input, filname))
            tot_samples += y.size(1)
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000).unsqueeze(0)

            codes = model.encode(y.to(device))
            audio = model.decode(codes).cpu().squeeze(0)

            output_file = os.path.join(a.output, os.path.splitext(filname)[0] + '_generated.wav')
            torchaudio.save(output_file, audio, 24000, encoding="PCM_S", bits_per_sample=16)
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
