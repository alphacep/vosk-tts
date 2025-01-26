import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class DeterministicDurationPredictor(nn.Module):
    def __init__(self, params):
        super().__init__()

    @torch.inference_mode()
    def forward(self, x, x_mask, temperature=1.0):
        return x * x_mask

    # StyleTTS duration
    def compute_loss(self, durations, enc_outputs, x_mask):
        max_phone_dur = 50

        estimations = enc_outputs.transpose(1, 2)

        durations = torch.clamp_max(durations, max_phone_dur-1).long().squeeze(1)
        x_mask = x_mask.squeeze(1)

        loss_dur = 0
        loss_ce = 0
        for dur, enc, mask in zip(durations, estimations, x_mask):
            dlen = sum(mask).long()
            enc = enc[:dlen, :]
            dur = dur[:dlen]
            trg = torch.zeros_like(enc)
            for p in range(trg.shape[0]):
                trg[p, :dur[p]] = 1
            dur_pred = torch.sigmoid(enc).sum(axis=1)
#            print ("Predicted", dur_pred)
#            print ("Target   ", dur)
            l1 = F.l1_loss(torch.log(dur_pred), torch.log(dur))
            ce = F.binary_cross_entropy_with_logits(enc.flatten(), trg.flatten())
#            print ("Dur loss   ", l1, ce)
            loss_dur += l1 
            loss_ce += ce

        loss_dur /= durations.size(0)
        loss_ce /= durations.size(0)
        print ("Dur loss L1", loss_dur)
        print ("Dur loss CE", loss_ce)
        return loss_dur + 20 * loss_ce
 




# Meta class to wrap all duration predictors


class DP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.name = params.name

        self.dp = DeterministicDurationPredictor(params)

    @torch.inference_mode()
    def forward(self, enc_outputs, mask, temperature=1.0):
        return self.dp(enc_outputs, mask, temperature=temperature)

    def compute_loss(self, durations, enc_outputs, mask):
        return self.dp.compute_loss(durations, enc_outputs, mask)
