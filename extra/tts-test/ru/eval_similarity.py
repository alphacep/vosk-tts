#!/usr/bin/env python3

import numpy as np
import sys
import os
import wespeaker

model = wespeaker.load_model_local("wespeaker/wespeaker-resnet34-LM")

def main(output_dir=""):
    scores = []
    files = [output_dir + "/" + f for f in sorted(os.listdir(output_dir))]
    for f in files:
        if not ".wav" in f:
            continue
        score = model.compute_similarity(f, f.replace(f"{output_dir}", "eval-ref"))
        print (f, score)
        scores.append(score)
    nscores = np.array(scores)
    print (f"Mean {np.mean(nscores):.3f} Min {np.min(nscores):.3f}")
    
main(sys.argv[1])
