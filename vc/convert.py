import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="logs/quickvc/config.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="logs/quickvc/quickvc.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="convert.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/quickvc", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = net_g.eval()
    total = sum([param.nelement() for param in net_g.parameters()])
 
    print("Number of parameter: %.2fM" % (total/1e6))
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None)

    print(f"Loading hubert_soft checkpoint")
    hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(device)
    print("Loaded soft hubert.")
    
    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")

    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(device)
            mel_tgt = mel_spectrogram_torch(
                wav_tgt, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).unsqueeze(0).to(device)
            print(wav_src.size())
            #long running
            #do something other
            c = hubert_soft.units(wav_src)

            
            
            c=c.transpose(2,1)
            #print(c.size())
            audio = net_g.infer(c, mel=mel_tgt)
         
            audio = audio[0][0].data.cpu().float().numpy()
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+title)), hps.data.sampling_rate, audio)
            else:
                write(os.path.join(args.outdir, f"{title}.wav"), hps.data.sampling_rate, audio)
            