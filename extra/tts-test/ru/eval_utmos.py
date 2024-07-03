#!/usr/bin/env python3

import utmos
import numpy as np
import sys
import os

model = utmos.Score() # The model will be automatically downloaded and will automatically utilize the GPU if available.

def main(output_dir=""):
    scores = []
    files = [output_dir + "/" + f for f in sorted(os.listdir(output_dir))]
    for f in files:
        score = model.calculate_wav_file(f)
        print (f, score)
        scores.append(score)
    nscores = np.array(scores)
    print (f"Mean {np.mean(nscores):.3f} Min {np.min(nscores):.3f}")
    
main(sys.argv[1])
