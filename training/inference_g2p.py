import librosa
import matplotlib.pyplot as plt

import os
import json
import math

import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence, text_to_sequence_g2p

import numpy as np
from scipy.io.wavfile import write
import re
from scipy import signal

## check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"

def get_text(text, hps):
    text_norm = text_to_sequence_g2p(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    print (text_norm)
    return text_norm

def vcss(out, inputstr, i): # single
    stn_tst = get_text(inputstr, hps)

    speed = 1.0
    output_dir = 'output0'
    sid = torch.LongTensor([i])
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
                0, 0].data.cpu().numpy() * 32768.0 # vol scale
        print (audio, np.max(audio))
    write(f'./{output_dir}/{out}.wav', hps.data.sampling_rate, audio.astype(np.int16))
    print(f'./{output_dir}/{out}.wav Generated!')


hps = utils.get_hparams_from_file("./configs/base.json")

if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

torch.set_printoptions(threshold=10_000)

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=58,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint("G.pth", net_g, None)

for i, line in enumerate(open("db/metadata.csv.test", encoding='utf-8')):
    items = line.split("|")
    vcss(items[0], items[1], 40)
