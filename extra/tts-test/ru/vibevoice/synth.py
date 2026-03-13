#!/usr/bin/env python3
import os
from timeit import default_timer as timer

import torch
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


MODEL_PATH = "VibeVoice-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000


# -------------------------
# Load processor + model
# -------------------------
print(f"Loading VibeVoice from {MODEL_PATH} on {DEVICE}")

processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)

model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE
)

model.eval()
model.set_ddpm_inference_steps(num_steps=10)


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
            text = "Speaker 1: " + items[1].replace("+", "")

            reffn, reftext = spkmap[spk]
            reftext = reftext.replace("+", "")
            reffn = os.path.join("eval-speakers-text/wav", reffn)

            # -------------------------
            # Prepare VibeVoice input
            # -------------------------
            inputs = processor(
                text=[text],                  # batch size 1
                voice_samples=[[reffn]],      # list of voices per sample
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(DEVICE)

            # -------------------------
            # Generate
            # -------------------------
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.3,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    is_prefill=True,  # voice cloning ON
                )

            audio = outputs.speech_outputs[0]

            output_path = os.path.join(out_dir, uid + ".wav")

            processor.save_audio(
                audio,
                output_path=output_path,
            )

            total_audio_samples += audio.shape[-1]

    end = timer()

    audio_duration_sec = total_audio_samples / SAMPLE_RATE
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
