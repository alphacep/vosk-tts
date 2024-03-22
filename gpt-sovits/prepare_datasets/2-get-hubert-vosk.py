import sys, os
sys.path.insert(0, "/home/ubuntu/storage/tts/GPT-SoVITS/GPT_SoVITS")
from feature_extractor import cnhubert
cnhubert.cnhubert_base_path = "pretrained_models/chinese-hubert-base"
import librosa, torch, torchaudio
from torchaudio.functional import resample

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = cnhubert.get_model()
model = model.to(device)

def extract_hubert(wav_path):
    print (wav_path)
    tensor_wav22, sr = torchaudio.load(wav_path)
    tensor_wav16 = resample(tensor_wav22, sr, 16000)
    tensor_wav16 = tensor_wav16.to(device)
    hubert = model.model(tensor_wav16)["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
    hubert_path = wav_path.replace(".wav", ".pt").replace("db/db", "db/cnhubert")
    hubert_folder = os.path.dirname(hubert_path)
    os.makedirs(hubert_folder, exist_ok=True)
    torch.save(hubert, hubert_path)

for line in open("db/metadata-phones-ids.csv"):
    wav_path, spk_name, text, aligned = line.split("|")
    extract_hubert(wav_path)
