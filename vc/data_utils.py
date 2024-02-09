import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
import utils
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text, transform
from torch.utils.data import DataLoader

"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length  = hparams.data.filter_length
        self.hop_length     = hparams.data.hop_length
        self.win_length     = hparams.data.win_length
        self.sampling_rate  = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.use_spk = hparams.model.use_spk
        self.spec_len = hparams.train.max_speclen

        random.seed(1243)
        random.shuffle(self.audiopaths)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        lengths = []
        for audiopath in self.audiopaths:
            lengths.append(os.path.getsize(audiopath[0]) // (2 * self.hop_length))
        self.lengths = lengths
    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        
        spec_filename = filename.replace(".wav", ".spec.pt")

        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
            
        

        #i = 80#random.randint(68,92)
  
        c_filename = filename.replace(".wav", f".npy")
        c = np.load(c_filename)#.squeeze(0)
        c=c.transpose(1,0)
        c = torch.FloatTensor(c)#.squeeze(0)
        
        return c, spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)
class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, hps):
        self.hps = hps
        self.use_sr = hps.train.use_sr
        self.use_spk = hps.model.use_spk

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)
        max_c_len = max([x[0].size(1) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        #print(max_spec_len,max_c_len)
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if self.use_spk:
            spks = torch.FloatTensor(len(batch), batch[0][3].size(0))
        else:
            spks = None

        c_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            if self.use_spk:
                spks[i] = row[3]
        #"""
        spec_seglen = spec_lengths[-1] if spec_lengths[-1] < self.hps.train.max_speclen + 1 else self.hps.train.max_speclen + 1
        #print(spec_seglen)
        wav_seglen = spec_seglen * self.hps.data.hop_length 
        #print(spec_padded.size(), spec_lengths, spec_seglen)
        spec_padded, ids_slice = commons.rand_spec_segments(spec_padded, spec_lengths, spec_seglen)
        wav_padded = commons.slice_segments(wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen)

        c_padded = commons.slice_segments(c_padded, ids_slice, spec_seglen)[:,:,:-1]

        spec_padded = spec_padded[:,:,:-1]
        wav_padded = wav_padded[:,:,:-self.hps.data.hop_length]
        #"""
        if self.use_spk:
          return c_padded, spec_padded, wav_padded, spks
        else:
          return c_padded, spec_padded, wav_padded


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

        self.buckets, self.boundaries, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        init_buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                init_buckets[idx_bucket].append(i)

#        for i in range(len(buckets) - 1, 0, -1):
#            if len(buckets[i]) == 0:
#                buckets.pop(i)
#                self.boundaries.pop(i+1)
         # collect non-zero buckets
        buckets = []
        boundaries = []
        for bucket, boundary in zip(init_buckets, self.boundaries):
            if len(bucket) != 0:
                buckets.append(bucket)
                boundaries.append(boundary)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, boundaries, num_samples_per_bucket

    def __iter__(self):
      # deterministically shuffle based on epoch
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

          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]

          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
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
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size


if __name__ == "__main__":
    hps = utils.get_hparams()
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,70,100,200,300,400,500,600,700,800,900,1000],
        num_replicas=1,
        rank=0,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate(hps)
    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)

    for batch_idx, (c, spec, y) in enumerate(train_loader):
        print(c.size(), spec.size(), y.size())
        #print(batch_idx, c, spec, y)
        #break
