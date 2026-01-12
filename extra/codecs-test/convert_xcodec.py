from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import glob
import os
import argparse
import torch
import torchaudio
from safetensors import safe_open
from timeit import default_timer as timer

from omegaconf import OmegaConf
from models.soundstream_semantic import SoundStream

def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model

device = 'cuda'

config = OmegaConf.load("xcodec/hubert_base_general_audio/config_hubert_general.yaml")
soundstream = build_codec_model(config)
parameter_dict = torch.load("xcodec/hubert_base_general_audio/xcodec_hubert_general_audio_v2.pth")
soundstream.load_state_dict(parameter_dict)  # Load model
soundstream = soundstream.to(device)
soundstream.eval()


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

count_parameters(soundstream)

def inference(a):
    filelist = os.listdir(a.input)
    os.makedirs(a.output, exist_ok=True)

    tot_samples = 0
    start = timer()

    with torch.no_grad():
        for i, filname in enumerate(filelist):

            y, sr = torchaudio.load(os.path.join(a.input, filname))
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=16000)
            tot_samples += y.size(1)

            wav = y.unsqueeze(1).cuda()
            compressed = soundstream.encode(wav, target_bw = 4)
            audio_out = soundstream.decode(compressed)
            audio_out = audio_out.detach().cpu().squeeze(0)

            output_file = os.path.join(a.output, os.path.splitext(filname)[0] + '_generated.wav')
            torchaudio.save(output_file, audio_out, sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
            print(output_file)

    end = timer()
    print (f"Processed {tot_samples / 16000.:.3f} seconds of speech in {end-start} seconds, xRT {(end-start) / (tot_samples / 16000.):3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='test_files')
    parser.add_argument('--output', default='generated_files')
    a = parser.parse_args()
    inference(a)

if __name__ == '__main__':
    main()
