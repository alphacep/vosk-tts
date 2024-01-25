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
        audio_norm = audio * max_wav_value * 3.0
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

    def synth_audio(self, text, speaker_id=0, noise_level=0.666667, speech_rate=1.0, duration_noise_level=0.8, scale=1.0):

        phoneme_ids = self.model.g2p(text)

        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)
        scales = np.array([noise_level, speech_rate, duration_noise_level], dtype=np.float32)

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
                "input": text,
                "input_lengths": text_lengths,
                "sid": sid,
                "scales": scales,
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

    def synth(self, text, oname, speaker_id=0, noise_level=0.666667, speech_rate=1.0, duration_noise_level=0.8):

        audio = self.synth_audio(text, speaker_id, noise_level, speech_rate, duration_noise_level)

        with wave.open(oname, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            f.writeframes(audio.tobytes())
