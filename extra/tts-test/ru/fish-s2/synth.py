#!/usr/bin/env python3
import os
from timeit import default_timer as timer

import torch
import torchaudio


from fish_speech.models.text2semantic.inference import init_model, generate_long

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16
llama_model, decode_one_token = init_model(
    checkpoint_path="/home/ubuntu/fish-speech/fishaudio-s2-pro",
    device=device,
    precision=precision,
    compile=True,
)

with torch.device(device):
    llama_model.setup_caches(
        max_batch_size=1,
        max_seq_len=llama_model.config.max_seq_len,
        dtype=next(llama_model.parameters()).dtype,
    )


def load_codec(codec_checkpoint_path, target_device, target_precision):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("/home/ubuntu/fish-speech/fish-speech/fish_speech/configs/modded_dac_vq.yaml")
    codec = instantiate(cfg)

    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=target_device, dtype=target_precision)
    return codec


codec_model = load_codec("/home/ubuntu/fish-speech/fishaudio-s2-pro/codec.pth", device, precision)

@torch.no_grad()
def encode_reference_audio(audio_path):
    wav, sr = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=codec_model.sample_rate)
    wav = resampler(wav)
    audios = wav.unsqueeze(0).to(precision).to('cuda')
    audio_lengths = torch.tensor([wav.shape[-1]], device=device, dtype=torch.long)  # time dimension
    indices, feature_lengths = codec_model.encode(audios, audio_lengths)
    return indices[0, :, :feature_lengths[0]]


@torch.no_grad()
def decode_codes_to_audio(merged_codes):
    audio = codec_model.from_indices(merged_codes[None])
    return audio[0, 0]


# -------------------------
spkmap = {}
with open("eval-speakers-text/metadata-phones-ids.csv.test-ref", encoding="utf-8") as f:
    for line in f:
        items = line.strip().split("|")
        spk = items[0].split("/")[0]
        spkmap[spk] = (items[0].replace("/", "_"), items[1])


# -------------------------
# Synthesis
# -------------------------
def main_synth(out_dir="out"):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_audio_samples = 0

    with open("eval.csv", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("|")
            fitems = items[0].split("/")

            uid = fitems[2] + "_" + fitems[-1][:-4]
            spk = items[0].split("/")[2]
            text = items[1].replace("+", "")

            reffn, reftext = spkmap[spk]
            reftext = reftext.replace("+", "")
            reffn = os.path.join("eval-speakers-text/wav", reffn)

            prompt_tokens_list = [encode_reference_audio(reffn).cpu()]

            generator = generate_long(
                model=llama_model,
                device=device,
                decode_one_token=decode_one_token,
                text=text,
                num_samples=1,
                max_new_tokens=1024,
                top_p=0.7,
                top_k=30,
                temperature=0.7,
                repetition_penalty=1.5,
                compile=False,
                iterative_prompt=True,
                chunk_length=200,
                prompt_text=[reftext],
                prompt_tokens=prompt_tokens_list,
            )

            codes = []
            for response in generator:
                if response.action == "sample":
                    codes.append(response.codes)
                elif response.action == "next":
                    break

            merged_codes = codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
            merged_codes = merged_codes.to(device)

            audio = decode_codes_to_audio(merged_codes)
            audio = (audio * 32768.0).to(torch.int16).cpu()
            
            total_audio_samples += audio.shape[-1]
            output_path = os.path.join(out_dir, uid + ".wav")
            torchaudio.save(output_path, audio.unsqueeze(0), codec_model.sample_rate)


    end = timer()

    audio_duration_sec = total_audio_samples / codec_model.sample_rate
    infer_sec = end - start
    real_time_factor = (
        infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
    )

    print(
        f"Real-time factor: {real_time_factor:.4f} "
        f"(infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)"
    )


if __name__ == "__main__":
    main_synth()
