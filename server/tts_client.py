#!/usr/bin/env python3

import argparse
import grpc
import wave

import tts_service_pb2
import tts_service_pb2_grpc

def run(text, oname):
    
    with wave.open(oname, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(22050)

        channel = grpc.insecure_channel('localhost:5001')
        stub = tts_service_pb2_grpc.SynthesizerStub(channel)
        it = stub.UtteranceSynthesis(tts_service_pb2.UtteranceSynthesisRequest(text=text))

        for r in it:
            f.writeframes(r.audio_chunk.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', required=True, help='text to synthesize')
    parser.add_argument('--out', required=True, help='path to store result')
    args = parser.parse_args()

    run(args.text, args.out)
