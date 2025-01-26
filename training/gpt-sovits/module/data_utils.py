import torchaudio
from torchaudio.functional import resample
import time
import logging
import os
import random
import traceback
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import re

from module import commons
from module.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text
import torch.nn.functional as F
from functools import lru_cache
import requests
from scipy.io import wavfile
from io import BytesIO

pattern = "([,.?!;:\"() ])"
def text_to_sequence_aligned(text):
    phonemes = []
    for word in re.split(pattern, text):
        if word == "":
            continue
        if "_" in word:
            phonemes.extend(word.split("_"))
        else:
            phonemes.append(word)

    sequence = cleaned_text_to_sequence(phonemes)
#    print (text, sequence)
    return sequence

# ZeroDivisionError fixed by Tybost (https://github.com/RVC-Boss/GPT-SoVITS/issues/79)
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, hparams, val=False):

        exp_dir = hparams.exp_dir
        audiopaths_sid_text = []

        if val:
            ext = "dev"
        else:
            ext = "train"

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.val = val

        for line in open(f"db/metadata-phones-ids.csv.{ext}"):
            items = line.strip().split("|")
            size = os.path.getsize(items[0])
            duration = size / self.sampling_rate / 2
            if duration > 20.0:
                continue
            items.append(size // (2 * self.hop_length)) #  length
            audiopaths_sid_text.append(items)

        random.seed(1234)
        random.shuffle(audiopaths_sid_text)

        print("wav_data_len:", len(audiopaths_sid_text))

        self.audiopaths_sid_text = [item[0:4] for item in audiopaths_sid_text]
        self.lengths = [item[4] for item in audiopaths_sid_text]

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, spk, text, aligned_text = audiopath_sid_text
        phoneme_ids = text_to_sequence_aligned(aligned_text)
        text = torch.FloatTensor(phoneme_ids)
        spec, wav = self.get_audio(audiopath)
        with torch.no_grad():
            ssl_path = audiopath.replace(".wav", ".pt").replace("db/db", "db/cnhubert")
            ssl = torch.load(ssl_path, map_location="cpu")
            if (ssl.shape[-1] != spec.shape[-1]):
                dtype = ssl.dtype
                ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(dtype)

            ssl.requires_grad = False
        return (ssl, spec, wav, text)

    def get_audio(self, filename):
        audio, sr = torchaudio.load(filename)
        audior = resample(audio, sr, self.sampling_rate)
        spec = spectrogram_torch(audior, self.filter_length, self.sampling_rate, self.hop_length, self.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        return spec, audior

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def random_slice(self, ssl, wav, mel):
        assert abs(ssl.shape[-1] - wav.shape[-1] // self.hop_length) < 3, (
        "first", ssl.shape, wav.shape)

        len_mel = mel.shape[1]
        if self.val:
            reference_mel = mel[:, :len_mel // 3]
            return reference_mel, ssl, wav, mel
        dir = random.randint(0, 1)
        sep_point = random.randint(int(len_mel // 3), int(len_mel // 3 * 2))

        if dir == 0:
            reference_mel = mel[:, :sep_point]
            ssl = ssl[:, :, sep_point:]
            wav2 = wav[:, sep_point * self.hop_length:]
            mel = mel[:, sep_point:]
        else:
            reference_mel = mel[:, sep_point:]
            ssl = ssl[:, :, :sep_point]
            wav2 = wav[:, :sep_point * self.hop_length]
            mel = mel[:, :sep_point]

        assert abs(ssl.shape[-1] - wav2.shape[-1] // self.hop_length) < 3, (
        ssl.shape, wav.shape, wav2.shape, mel.shape, sep_point, self.hop_length, sep_point * self.hop_length, dir)
        return reference_mel, ssl, wav2, mel


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_ssl_len = max([x[0].size(2) for x in batch])
        max_ssl_len = int(2 * ((max_ssl_len // 2) + 1))
        max_spec_len = max([x[1].size(1) for x in batch])
        max_spec_len = int(2 * ((max_spec_len // 2) + 1))
        max_wav_len = max([x[2].size(1) for x in batch])
        max_text_len = max([x[3].size(0) for x in batch])

        ssl_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        ssl_padded = torch.FloatTensor(len(batch), batch[0][0].size(1), max_ssl_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)

        spec_padded.zero_()
        wav_padded.zero_()
        ssl_padded.zero_()
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            ssl = row[0]
            ssl_padded[i, :, :ssl.size(2)] = ssl[0, :, :]
            ssl_lengths[i] = ssl.size(2)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            text = row[3]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

        return ssl_padded, ssl_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, text_padded, text_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size