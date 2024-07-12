#!/usr/bin/env python3

import sys
import os
import json
import math
import sherpa_onnx
import scipy
from timeit import default_timer as timer
import numpy as np

def main(output_dir=""):

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder="vosk-model-ru/am-onnx/encoder.onnx",
            decoder="vosk-model-ru/am-onnx/decoder.onnx",
            joiner="vosk-model-ru/am-onnx/joiner.onnx",
            tokens="vosk-model-ru/lang/tokens.txt",
            provider="cpu",
            num_threads=8,
            sample_rate=16000,
            decoding_method="greedy_search")

    start_time = timer()
    n_samples = 0

    batch_size = 8
    files = [output_dir + "/" + f for f in sorted(os.listdir(output_dir))]
    for i in range(0, len(files), batch_size):
        streams = []
        batch = files[i:i+batch_size]
        for wave_filename in batch:
            sample_rate, samples = scipy.io.wavfile.read(wave_filename)
            s = recognizer.create_stream()
            s.accept_waveform(sample_rate, samples / 32768)
            n_samples += len(samples)
            tail_paddings = np.random.normal(0, 1, int(0.4 * sample_rate)) / 32768
            s.accept_waveform(sample_rate, tail_paddings)
            streams.append(s)
        recognizer.decode_streams(streams)
        results = [s.result.text for s in streams]
        for f, result in zip(batch, results):
            print (f.split("/")[-1][0:-4], result.lower().strip())

    end_time = timer()

    print("Processed %.3f seconds of audio in %.3f seconds (%.3f xRT)"
        % (n_samples / 16000.0,
        end_time - start_time,
        (end_time - start_time) / (n_samples / 16000.0)),
        file=sys.stderr)

main(sys.argv[1])
