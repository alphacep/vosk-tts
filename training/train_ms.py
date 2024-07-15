import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import tqdm
from pqmf import PQMF
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler
)
from models import (
    SynthesizerTrn,
    MultiPeriodMultiSpecDiscriminator,
    DurationDiscriminator,
    DurationDiscriminator2,
    AVAILABLE_FLOW_TYPES,
    AVAILABLE_DURATION_DISCRIMINATOR_TYPES,
    WavLMDiscriminator
)
from losses import (
    generator_loss,
    discriminator_loss,
    generator_TPRLS_loss,
    discriminator_TPRLS_loss,
    feature_loss,
    kl_loss,
    subband_stft_loss,
    WavLMLoss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6060'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    net_dur_disc = None
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    if os.name == 'nt':
        dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)
    else:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=hps.train.batch_size, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)
    # some of these flags are not being used in the code and directly set in hps json file.
    # they are kept here for reference and prototyping.
    if "use_transformer_flows" in hps.model.keys() and hps.model.use_transformer_flows == True:
        use_transformer_flows = True
        transformer_flow_type = hps.model.transformer_flow_type
        print(f"Using transformer flows {transformer_flow_type} for VITS2")
        assert transformer_flow_type in AVAILABLE_FLOW_TYPES, f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
    else:
        print("Using normal flows for VITS1")
        use_transformer_flows = False

    if "use_spk_conditioned_encoder" in hps.model.keys() and hps.model.use_spk_conditioned_encoder == True:
        if hps.data.n_speakers == 0:
            raise ValueError("n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model")
        use_spk_conditioned_encoder = True
    else:
        print("Using normal encoder for VITS1")
        use_spk_conditioned_encoder = False

    if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas == True:
        print("Using noise scaled MAS for VITS2")
        use_noise_scaled_mas = True
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        use_noise_scaled_mas = False
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    if "use_duration_discriminator" in hps.model.keys() and hps.model.use_duration_discriminator == True:
        # print("Using duration discriminator for VITS2")
        use_duration_discriminator = True

        #- for duration_discriminator2
        # duration_discriminator_type = getattr(hps.model, "duration_discriminator_type", "dur_disc_1")
        duration_discriminator_type = hps.model.duration_discriminator_type
        print(f"Using duration_discriminator {duration_discriminator_type} for VITS2")
        assert duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys(), f"duration_discriminator_type must be one of {list(AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys())}"
        #DurationDiscriminator = AVAILABLE_DURATION_DISCRIMINATOR_TYPES[duration_discriminator_type]

        if duration_discriminator_type == "dur_disc_1":
            net_dur_disc = DurationDiscriminator(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)
        elif duration_discriminator_type == "dur_disc_2":
            net_dur_disc = DurationDiscriminator2(
                hps.model.hidden_channels,
#                hps.model.hidden_channels,
                256,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)

        '''
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
        '''
    else:
        print("NOT using any duration discriminator like VITS1")
        net_dur_disc = None
        use_duration_discriminator = False

    if ("use_wd" in hps.model.keys() and hps.model.use_wd):
        net_wd = WavLMDiscriminator(
            hps.model.slm_hidden,
            hps.model.slm_nlayers,
            hps.model.slm_initial_channel
        ).cuda(rank)

        wl = WavLMLoss(
            hps.model.slm_model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm_sr,
        ).to(rank)
        print("Using WavLMDiscriminator")
    else:
        net_wd = None
        wl = None

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model).cuda(rank)
    net_d = MultiPeriodMultiSpecDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps)
    else:
        optim_dur_disc = None

    if net_wd:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_wd = None

#    for param in net_d.parameters():
#        param.requires_grad = False
#    for param in net_g.parameters():
#        param.requires_grad = False
#    for param in net_g.dp.parameters():
#        param.requires_grad = True
#    for param in net_dur_disc.parameters():
#        param.requires_grad = False
#    for param in net_wd.parameters():
#        param.requires_grad = False

#    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
#    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

#    if net_dur_disc is not None:  # 2의 경우
#        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)

#    if net_wd:
#        net_wd = DDP(
#            net_wd,
#            device_ids=[rank],
#            find_unused_parameters=True
#        )

    if True:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g)
        _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d)
        if net_dur_disc is not None:
            _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                                                       net_dur_disc, optim_dur_disc)

        if net_wd:
            _, _, _, _ = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth"),
                net_wd,
                optim_wd,
            )

        global_step = (epoch_str - 1) * len(train_loader)
    else:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(optim_dur_disc, gamma=hps.train.lr_decay,
                                                                    last_epoch=epoch_str - 2)
    else:
        scheduler_dur_disc = None

    if net_wd:
        scheduler_wd = torch.optim.lr_scheduler.ExponentialLR(
            optim_wd, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_wd = None

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_dur_disc, wl, net_wd], [optim_g, optim_d, optim_dur_disc, optim_wd],
                               [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd], scaler, [train_loader, eval_loader],
                               logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_dur_disc, wl, net_wd], [optim_g, optim_d, optim_dur_disc, optim_wd],
                               [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()
        if net_wd:
            scheduler_wd.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d, net_dur_disc, wl, net_wd = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    if net_wd:
        net_wd.train()

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc='Loading train data')
    else:
        loader = train_loader
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(loader):
        if net_g.use_noise_scaled_mas:
            current_mas_noise_scale = net_g.mas_noise_scale_initial - net_g.noise_scale_delta * global_step
            net_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_) = net_g(x, x_lengths, spec, spec_lengths,
                                                                                speakers)

            if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
                mel = spec
            else:
                mel = spec_to_mel_torch(
                    # spec,
                    spec.float(), #- for 16bit stability
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_tprls = discriminator_TPRLS_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc + loss_disc_tprls

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), x_mask.detach(), logw_.detach(),
                                                        logw.detach())
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

            if wl is not None and net_wd is not None:
                with autocast(enabled=False):
                   loss_slm = wl.discriminator(
                       y.detach().squeeze(), y_hat.detach().squeeze()
                   ).mean()
                optim_wd.zero_grad()
                scaler.scale(loss_slm).backward()
                scaler.unscale_(optim_wd)
                grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
                scaler.step(optim_wd)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_r, y_d_hat_g)
                loss_gen_tprls = generator_TPRLS_loss(y_d_hat_r, y_d_hat_g)

                if hps.model.mb_istft_vits == True:
                    pqmf = PQMF(y.device)
                    y_mb = pqmf.analysis(y)
                    loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
                else:
                    loss_subband = torch.tensor(0.0)

                loss_gen_all = loss_gen + loss_gen_tprls + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_r,  y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

                if net_wd is not None:
                    loss_lm = wl(y.detach().squeeze(), y_hat.squeeze()).mean()
                    loss_lm_gen = wl.generator(y_hat.squeeze())
                    loss_gen_all += loss_lm
                    loss_gen_all += loss_lm_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_subband]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                if net_dur_disc is not None:
                    scalar_dict.update(
                        {"loss/dur_disc/total": loss_dur_disc_all, "grad_norm_dur_disc": grad_norm_dur_disc})
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl,
                     "loss/g/subband": loss_subband})

                scalar_dict.update({"loss/d/tprls": loss_disc_tprls, "loss/g/tprls": loss_gen_tprls})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                if net_dur_disc is not None:
                   scalar_dict.update({"loss/dur_disc_r/{}".format(i): v for i, v in enumerate(losses_dur_disc_r)})
                   scalar_dict.update({"loss/dur_disc_g/{}".format(i): v for i, v in enumerate(losses_dur_disc_g)})
                   scalar_dict.update({"loss/dur_gen" : loss_dur_gen})

                if net_wd:
                    scalar_dict.update(
                        {
                            "loss/wd/total": loss_slm.item(),
                            "loss/wd/lm": loss_lm.item(),
                            "loss/wd/lm_gen": loss_lm_gen.item(),
                            "grad_norm_wd": grad_norm_wd,
                        }
                    )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))

#                utils.save_checkpoint(net_g.dp, optim_g, hps.train.learning_rate, epoch,
#                                      os.path.join(hps.model_dir, "DP_{}.pth".format(global_step)))

                if net_dur_disc is not None:
                    utils.save_checkpoint(net_dur_disc, optim_dur_disc, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)))

                if net_wd:
                    utils.save_checkpoint(
                        net_wd,
                        optim_wd,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "WD_{}.pth".format(global_step)))

#                prev_g = os.path.join(hps.model_dir, "G_{}.pth".format(global_step - 3 * hps.train.eval_interval))
#                if os.path.exists(prev_g):
#                    os.remove(prev_g)
#                    prev_d = os.path.join(hps.model_dir, "D_{}.pth".format(global_step - 3 * hps.train.eval_interval))
#                    if os.path.exists(prev_d):
#                        os.remove(prev_d)
#                        prev_dur = os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step - 3 * hps.train.eval_interval))
#                        if os.path.exists(prev_dur):
#                            os.remove(prev_dur)

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            break
        y_hat, y_hat_mb, attn, mask, *_ = generator.infer(x, x_lengths, speakers, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
        "gen/audio": y_hat[0, :, :y_hat_lengths[0]]
    }
    if global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    main()
