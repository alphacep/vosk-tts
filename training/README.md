Training code for multispeaker Russian MB-iSTFT-VITS2 models

Code mostly from <https://github.com/FENRlR/MB-iSTFT-VITS2>

It is recommended to start from pretrained checkpoint https://huggingface.co/alphacep/vosk-tts-ru-multi

## Finetuning for your own voice

Here are the steps to run finetuning:

0) Clone this repository and build monotonic align

```
git clone https://github.com/alphacep/vosk-tts
cd vosk-tts/training
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

1) clone the pretrained model

```
git clone https://huggingface.co/alphacep/vosk-tts-ru-multi pretrained
```

2) get a dictionary from existing model and put it into db/dictionary

```
wget https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.7-multi.zip
unzip vosk-model-tts-ru-0.7-multi.zip
mkdir -p db
cp vosk-model-tts-ru-0.7-multi/dictionary db/dictionary
```

3) Clone SLM


```
git clone https://huggingface.co/microsoft/wavlm-base-plus
```

4) download the existing voice adaptation pack

```
wget https://alphacephei.com/tts/db-finetune.zip
unzip db-finetune.zip
```

If you prepare your own data, consider the following points:

  1. Audio must be 22050Hz mono strictly
  1. Transcription in metadata.csv must strictly match the audio
  1. You can prepare metadata.csv with python script if your input data has different format
  1. The more data you have the better but it is ok to start with 50 utterances
  1. You need to add missing words to db/dictionary

5) Run the training. It requires a GPU card and takes about 40 minutes

```
python3 train_finetune.py
```

6) If you have significant amount of data, enable discriminator training otherwise just update the generator here:

<https://github.com/alphacep/vosk-tts/blob/master/training/train_finetune.py#L220>

Overall, the more data you have the more you can train to get best quality.

7) Export the model

```
python3 export_onnx.py
```

8) Copy the model into existing voice and test it

```
mv model.onnx vosk-model-ru-0.7-multi
vosk-tts -m vosk-model-ru-0.7-multi -s 0 -i "Привет!" -o out.wav
aplay out.wav
```
