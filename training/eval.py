#!/usr/bin/env python3

import sys
import os
import json
import math

import requests
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence, text_to_sequence_g2p, text_to_sequence_aligned

import numpy as np
from scipy.io.wavfile import write
import re
from scipy import signal

import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
import sherpa_onnx
import wave


checkpoint_id = 2953

## check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"

def get_text(text, hps):
    text_norm = text_to_sequence_aligned(text)
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def synth(net_g, out, inputstr, i): # single

    hps = utils.get_hparams_from_file("./configs/base.json")

    output_dir = f"eval.{checkpoint_id}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with torch.no_grad():
        stn_tst = get_text(inputstr, hps)
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([i]).to(device)

        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)[0]
        audio = audio[0, 0].data.cpu().numpy() * 32768.0 # vol scale

    write(f'./{output_dir}/{out}.wav', hps.data.sampling_rate, audio.astype(np.int16))

def main_synth():
    hps = utils.get_hparams_from_file("./configs/base.json")

    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=58,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(f"logs/G_{checkpoint_id}000.pth", net_g, None)

    for i, line in enumerate(open("db/metadata-phones-ids.csv.test.full", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        synth(net_g, uid, items[3], int(items[1]))
    
main_synth()


def read_wave(wave_filename):
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def main():

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder="vosk-model-ru/am-onnx/encoder.onnx",
            decoder="vosk-model-ru/am-onnx/decoder.onnx",
            joiner="vosk-model-ru/am-onnx/joiner.onnx",
            tokens="vosk-model-ru/lang/tokens.txt",
            provider="cuda",
            num_threads=10,
            sample_rate=16000,
            decoding_method="greedy_search")

    start_time = timer()
    n_samples = 0

    batch_size = 8
    output_dir = f"eval.{checkpoint_id}"
    files = [output_dir + "/" + f for f in sorted(os.listdir(output_dir))]
    for i in range(0, len(files), batch_size):
        streams = []
        batch = files[i:i+batch_size]
        for wave_filename in batch:
            samples, sample_rate = read_wave(wave_filename)
            s = recognizer.create_stream()
            s.accept_waveform(sample_rate, samples)
            n_samples += len(samples)
            streams.append(s)
        recognizer.decode_streams(streams)
        results = [s.result.text for s in streams]
        for f, result in zip(batch, results):
            print (f.split("/")[-1][0:-4], result.lower().strip())

    end_time = timer()

    print("Processed %.3f seconds of audio in %.3f seconds (%.3f xRT)"
        % (n_samples / 16000.0,
        end_time - start_time,
        (end_time - start_time) / (n_samples / 16000.0)),
        file=sys.stderr)

main()


def dump_ref():

    punct_line = "\":?,.;!()[]_/|#=*^<>~&%"
    extra_punct = "“”„‚¿¡…•·∙→←،٫«»؟؛−–—©↑§×½±●"
    punct_line = punct_line + extra_punct
    nopunct = str.maketrans(punct_line, " " * len(punct_line))

    for i, line in enumerate(open("db/metadata-phones-ids.csv.test.full", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]

        text = items[2].lower().translate(nopunct)

        text = text.replace("+", "")
        text = " ".join([w for w in text.split() if w != '-'])
        print (uid, text)

#dump_ref()

def main_orig():

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder="vosk-model-ru/am-onnx/encoder.onnx",
            decoder="vosk-model-ru/am-onnx/decoder.onnx",
            joiner="vosk-model-ru/am-onnx/joiner.onnx",
            tokens="vosk-model-ru/lang/tokens.txt",
            provider="cuda",
            num_threads=10,
            sample_rate=16000,
            decoding_method="greedy_search")

    start_time = timer()
    n_samples = 0

    files = []
    for i, line in enumerate(open("db/metadata-phones-ids.csv.test.full", encoding='utf-8')):
        items = line.strip().split("|")
        files.append(items[0])

    batch_size = 8
    for i in range(0, len(files), batch_size):
        streams = []
        batch = files[i:i+batch_size]
        for wave_filename in batch:
            samples, sample_rate = read_wave(wave_filename)
            s = recognizer.create_stream()
            s.accept_waveform(sample_rate, samples)
            n_samples += len(samples)
            streams.append(s)
        recognizer.decode_streams(streams)
        results = [s.result.text for s in streams]
        for f, result in zip(batch, results):
            fitems = f.split("/")
            uid = fitems[2] + "_" + fitems[-1][:-4]
            print (uid, result.lower().strip())

    end_time = timer()

    print("Processed %.3f seconds of audio in %.3f seconds (%.3f xRT)"
        % (n_samples / 16000.0,
        end_time - start_time,
        (end_time - start_time) / (n_samples / 16000.0)),
        file=sys.stderr)

#main_orig()

def cp_ref():
    for i, line in enumerate(open("db/metadata-phones-ids.csv.test.full", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        os.system(f"cp {items[0]} eval-ref/{uid}.wav")

#cp_ref()
