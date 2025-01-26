import os

import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from random import shuffle
import logging, librosa, utils, torch
from module.models import SynthesizerTrn

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
hps = utils.get_hparams_from_file("configs/s2.json")
vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
)
current_model_dict = vq_model.state_dict()
loaded_state_dict = torch.load("pretrained_models/s2G488k.pth", map_location=torch.device('cpu'))["weight"]
new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
vq_model.load_state_dict(new_state_dict, strict=False)

vq_model = vq_model.to(device)
vq_model.eval()

def wav2sem(wav_path):
    hubert_path = wav_path.replace(".wav", ".pt").replace("db/db", "db/cnhubert")
    if not os.path.exists(hubert_path):
        return
    ssl_content = torch.load(hubert_path, map_location="cpu")
    ssl_content = ssl_content.to(device)
    codes = vq_model.extract_latent(ssl_content)
    semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
    return semantic

with open("db/semantic.csv", "w") as semfd:
    with open("db/metadata-phones-ids.csv.train", "r", encoding="utf8") as f:
        for line in f:
            wav_path = line.strip().split("|")[0]
            sem = wav2sem(wav_path)
            semfd.write(f"{wav_path}\t{sem}\n")
