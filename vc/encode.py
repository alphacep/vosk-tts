import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

import torch
import torchaudio
from torchaudio.functional import resample

def encode_dataset(args):
    contentvec_extractor = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best").cuda().eval()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if True:#not os.path.exists(out_path.with_suffix(".npy")):
            wav, sr = torchaudio.load(in_path)
            wav = resample(wav, sr, 16000)
            wav = wav.cuda()

            with torch.no_grad():
                 units = contentvec_extractor.extract(wav)
                 np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".wav",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
