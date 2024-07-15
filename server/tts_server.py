#!/usr/bin/env python3
#
# Copyright 2023 Alpha Cephei Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC TTS server."""

from concurrent import futures
import os
import sys
import time
import math
import logging
import grpc
import time

import tts_service_pb2
import tts_service_pb2_grpc

from vosk_tts import Model, Synth

vosk_interface = os.environ.get('VOSK_SERVER_INTERFACE', '0.0.0.0')
vosk_port = int(os.environ.get('VOSK_SERVER_PORT', 5001))
vosk_model_path = os.environ.get('VOSK_MODEL_PATH', 'vosk-model-tts-ru-0.7-multi')
vosk_threads = int(os.environ.get('VOSK_SERVER_THREADS', os.cpu_count() or 1))

class SynthesizerServicer(tts_service_pb2_grpc.SynthesizerServicer):
    def __init__(self):
        self.model = Model(model_path=vosk_model_path)
        self.synth = Synth(self.model)

    def UtteranceSynthesis(self, request, context):

        speaker_id = 0
        speech_rate = 1.0

        for hint in request.hints:
            if hint.HasField("speaker_id"):
                speaker_id = hint.speaker_id
            if hint.HasField("speech_rate"):
                speech_rate = int(hint.speech_rate)

        audio = self.synth.synth_audio(request.text, speaker_id=speaker_id, speech_rate=speech_rate)
        yield tts_service_pb2.UtteranceSynthesisResponse(audio_chunk=tts_service_pb2.AudioChunk(data=audio.tobytes()))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(vosk_threads))
    tts_service_pb2_grpc.add_SynthesizerServicer_to_server(SynthesizerServicer(), server)

    server.add_insecure_port(f"{vosk_interface}:{vosk_port}")
    server.start()
    logging.info(f"Listening on {vosk_interface}:{vosk_port}")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
