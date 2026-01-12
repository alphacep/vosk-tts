#!/usr/bin/env python3
import sys
import os
import torchaudio
import numpy as np
from pesq import pesq

def main(output_dir=""):
    scores = []
    files = [output_dir + "/" + f for f in sorted(os.listdir(output_dir))]
    for f in files:
        if ".wav" not in f:
            continue
        ref, sr = torchaudio.load(f.replace(output_dir, "wavs_test_ru").replace("_generated", ""))
        ref = torchaudio.functional.resample(ref, orig_freq=sr, new_freq=16000)
        deg, sr = torchaudio.load(f)
        deg = torchaudio.functional.resample(deg, orig_freq=sr, new_freq=16000)
        score = pesq(16000, ref.squeeze().numpy(), deg.squeeze().numpy(), 'wb')
        print (f, score)
        scores.append(score)
    nscores = np.array(scores)
    print (f"Mean {np.mean(nscores):.3f} Min {np.min(nscores):.3f}")
    
main(sys.argv[1])
