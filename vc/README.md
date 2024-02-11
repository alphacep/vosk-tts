# This is an extention of standard VITS-based VC

The codebase is based on [QuickVC](https://github.com/quickvc/QuickVC-VoiceConversion) but contains several modifications

1. TPRLS GAN loss (from StyleTTS2)
2. Multispectral GAN discriminator (Univnet/Vocos/StyleTTS2)
3. Contentvec instead of Hubert

## Pretrained model

Pretrained model is available on hugginface:

https://huggingface.co/alphacep/vosk-vc-ru

## Results

On Russian dataset we measure speaker similarity with Resemblyzer

|Model                                 | Average similarity | Min similarity |
|--------------------------------------|--------------------|----------------|
| **Our**                                                                    |
|Original QuickVC (trained on VCTK)    |            0.667   | 0.477          |
|Trained on Russian data               |            0.836   | 0.692          |
|With contentvec                       |            0.880   | 0.712          |
| **Others**                                                                 |
|Openvoice EN                          |            0.800   | 0.653          |

## TODO

  - [ ] Test other VC methods (XTTS, GPT-Sovits, RVC, Unitspeech)
  - [ ] Collect wideband dataset (currently 16khz)
  - [ ] Add better speaker and style encoder (3dspeaker, Openvoice)

## Inference with pretrained model

```python
python convert.py
```
You can change convert.txt to select the target and source

## Preprocess

```python
python encode.py dataset/VCTK-16K dataset/VCTK-16K
```

## Train

```python
python train.py
```

## References

Initial approach [QuickVC](https://github.com/quickvc/QuickVC-VoiceConversion)

Better content/speaker decomposition [Contentvec](https://github.com/auspicious3000/contentvec)

Fast MB-iSTFT decoder for VITS [MS-ISTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)

Hubert-soft [Soft-VC](https://github.com/bshall/hubert)

Data augmentation (not implemented) [FreeVC](https://github.com/OlaWod/FreeVC)

TPRLS GAN [StyleTTS2](https://github.com/yl4579/StyleTTS2), [Paper](https://dl.acm.org/doi/abs/10.1145/3573834.3574506)

Multires spectral discriminator [UnivNet](https://arxiv.org/abs/2106.07889)

