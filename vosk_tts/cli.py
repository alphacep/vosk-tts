#!/usr/bin/env python3

import argparse
import logging
import sys
import os

from .model import list_models, list_languages
from .model import Model
from .synth import Synth

parser = argparse.ArgumentParser(
        description = "Synthesize input")
parser.add_argument(
        "--model", "-m", type=str,
        help="model path")
parser.add_argument(
        "--list-models", default=False, action="store_true",
        help="list available models")
parser.add_argument(
        "--list-languages", default=False, action="store_true",
        help="list available languages")
parser.add_argument(
        "--model-name", "-n", type=str,
        help="select model by name")
parser.add_argument(
        "--lang", "-l", default="en-us", type=str,
        help="select model by language")
parser.add_argument(
        "--input", "-i", type=str,
        help="input string")
parser.add_argument(
        "--speaker", "-s", type=int,
        help="speaker id for multispeaker model")
parser.add_argument(
        "--output", "-o", default="out.wav", type=str,
        help="optional output filename path")
parser.add_argument(
        "--log-level", default="INFO",
        help="logging level")

def main():

    args = parser.parse_args()
    log_level = args.log_level.upper()
    logging.getLogger().setLevel(log_level)

    if args.list_models is True:
        list_models()
        return

    if args.list_languages is True:
        list_languages()
        return

    if not args.input:
        logging.info("Please specify input text or file")
        sys.exit(1)

    model = Model(args.model, args.model_name, args.lang)
    synth = Synth(model)
    synth.synth(args.input, args.output, args.speaker)

if __name__ == "__main__":
    main()
