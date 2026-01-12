#!/bin/bash

python firefly/vocoder/fish_vocoder/test.py task_name=firefly-gan-test \
    model/generator=firefly-gan-base \
    ckpt_path=firefly/firefly-gan-base.ckpt \
    input_path=$1 \
    output_path=$2
