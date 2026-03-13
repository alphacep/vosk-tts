#!/usr/bin/env python3
import os
from timeit import default_timer as timer

# -------------------------
# Load processor + model
# -------------------------
from pathlib import Path
import importlib.util
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


pretrained_model_name_or_path = "/home/ubuntu/moss/MOSS-TTS-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to('cpu')


model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    # If FlashAttention 2 is installed, you can set attn_implementation="flash_attention_2"
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()


# -------------------------
# Load speaker map
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


            input_text = [processor.build_user_message(text=text, reference=[reffn])]
            batch = processor(input_text, mode="generation")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
            )

            message = processor.decode(outputs)[0]
            audio = message.audio_codes_list[0]

            total_audio_samples += audio.shape[-1]
            output_path = os.path.join(out_dir, uid + ".wav")
            torchaudio.save(output_path, audio.unsqueeze(0), processor.model_config.sampling_rate)


    end = timer()

    audio_duration_sec = total_audio_samples / processor.model_config.sampling_rate
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
