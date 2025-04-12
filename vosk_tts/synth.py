import json
import numpy as np
import onnxruntime
import wave
import time
import logging

class Synth:

    def __init__(self, model):
        self.model = model
        self.multi = model.config["num_speakers"] > 1

    def audio_float_to_int16(self,
        audio: np.ndarray, max_wav_value: float = 32767.0
    ) -> np.ndarray:
        """Normalize audio and convert to int16 range"""
        audio_norm = audio * max_wav_value
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

    def synth_audio(self, text, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):

        if noise_level is None:
            noise_level = self.model.config["inference"].get("noise_level", 0.66667)
        if speech_rate is None:
            speech_rate = self.model.config["inference"].get("speech_rate", 1.0)
        if duration_noise_level is None:
            duration_noise_level = self.model.config["inference"].get("duration_noise_level", 0.8)
        if scale is None:
            scale = self.model.config["inference"].get("scale", 1.0)

        tokens = self.model.tokenizer.encode(text)
        bert = self.model.bert_onnx.run(
            None,
            {
               "input_ids": [tokens.ids],
               "attention_mask": [tokens.attention_mask],
               "token_type_ids": [tokens.type_ids],
            }
        )

        # Select only first token in multitoken words
        selected = [0]
        for i, t in enumerate(tokens.tokens):
            if t[0] != '#':
                selected.append(i)
        bert = bert[0][selected]

        phoneme_ids, bert_embs = self.model.g2p(text, bert)

        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)
        scales = np.array([noise_level, duration_noise_level, 1.0 / speech_rate], dtype=np.float32)
        bert_embs = np.expand_dims(np.transpose(np.array(bert_embs, dtype=np.float32)), 0)

        if self.multi:
            # Assign first voice
            if speaker_id is None:
                speaker_id = 0
            sid = np.array([speaker_id], dtype=np.int64)
        else:
            sid = None

        start_time = time.perf_counter()
        audio = self.model.onnx.run(
            None,
            {
                "x": text,
                "x_lengths": text_lengths,
                "scales": scales,
                "spks": sid,
                "bert": bert_embs,
            },
        )[0]
        audio = audio.squeeze()
        audio = audio * scale

        audio = self.audio_float_to_int16(audio)
        end_time = time.perf_counter()

        audio_duration_sec = audio.shape[-1] / 22050
        infer_sec = end_time - start_time
        real_time_factor = (
            infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        )

        logging.info("Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)" % (real_time_factor, infer_sec, audio_duration_sec))
        return audio

    def synth(self, text, oname, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):

        audio = self.synth_audio(text, speaker_id, noise_level, speech_rate, duration_noise_level, scale)

        with wave.open(oname, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            f.writeframes(audio.tobytes())
