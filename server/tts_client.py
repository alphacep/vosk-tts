#!/usr/bin/env python3

import argparse
import grpc
import wave

import tts_service_pb2
import tts_service_pb2_grpc

def run(text, oname, speaker, rate):
    
    with wave.open(oname, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(22050)

        channel = grpc.insecure_channel('localhost:5001')
        stub = tts_service_pb2_grpc.SynthesizerStub(channel)
        hints = [tts_service_pb2.Hints(speaker_id=speaker), tts_service_pb2.Hints(speech_rate=rate)]
        it = stub.UtteranceSynthesis(tts_service_pb2.UtteranceSynthesisRequest(text=text, hints=hints))

        for r in it:
            f.writeframes(r.audio_chunk.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='text to synthesize')
    parser.add_argument('--output', '-o', required=True, help='path to store result')
    parser.add_argument('--speaker', '-s', type=int, default=0, help='speaker id for multispeaker model')
    parser.add_argument('--speech-rate', '-r', type=float, default=1.0, help='speech rate of the synthesis')

    args = parser.parse_args()

    run(args.input, args.output, args.speaker, args.speech_rate)
