import json
import numpy as np
import onnxruntime
import wave
import time

class Synth:

    def __init__(self, model):
        self.model = model

    def audio_float_to_int16(self,
        audio: np.ndarray, max_wav_value: float = 32767.0
    ) -> np.ndarray:
        """Normalize audio and convert to int16 range"""
        audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

    def synth(self, text, oname):

        phoneme_ids = self.model.g2p(text)

        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

        start_time = time.perf_counter()
        audio = self.model.onnx.run(
            None,
            {
                "input": text,
                "input_lengths": text_lengths,
                "scales": scales,
            },
        )[0].squeeze((0, 1))

        audio = self.audio_float_to_int16(audio.squeeze())
        end_time = time.perf_counter()

        audio_duration_sec = audio.shape[-1] / 22050
        infer_sec = end_time - start_time
        real_time_factor = (
            infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        )

        print("Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)" % (real_time_factor, infer_sec, audio_duration_sec))

        with wave.open(oname, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            f.writeframes(audio.tobytes())
