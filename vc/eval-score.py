#!/usr/bin/env python3

import os
import argparse
import torch
import librosa
import time
import numpy as np
from tqdm import tqdm

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import torch.autograd.profiler as profiler

## check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"


from resemblyzer import preprocess_wav, VoiceEncoder

encoder = VoiceEncoder()

def compute_embedding(fpath):
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    return embed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="logs/quickvc/config.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="logs/quickvc/quickvc.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="dataset/eval.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="eval", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    

    allfiles, titles, srcs, tgts = [], [], [], []
    allfiles = open(args.txtpath, "r").readlines()

    for i in range(len(allfiles) // 2):
        tgt = allfiles[i].strip()
        src = allfiles[-i].strip()
        title = src.split("/")[-1][:-4] + "_" + tgt.split("/")[-1][:-4]
        titles.append(title)
        srcs.append(src)
        tgts.append(tgt)

    allscore = 0
    scores = []
    for tgt, res in zip(tgts, titles):
        tgt_emb = compute_embedding(tgt)
        res_emb = compute_embedding(f"{args.outdir}/{res}.wav")
        score = np.dot(tgt_emb, res_emb)
        scores.append(score)
        print (f"{tgt} {res} {score}")

    ns = np.array(scores)
    print (f"Average: {np.mean(ns)} Min: {np.min(ns)}")

