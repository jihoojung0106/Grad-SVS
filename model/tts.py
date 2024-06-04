# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch

# from model.monotonic_align import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder,PhonemesEncoder,NotesEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
import numpy as np

def calc_paths(log_p, token_lengths, mel_lengths):
    return np.stack([
        calc_path(x, token_Length, mel_Length)
        for x, token_Length, mel_Length in zip(log_p, token_lengths, mel_lengths)
        ], axis= 0)

def calc_path(x, token_length, mel_length):
    path = np.zeros_like(x).astype(np.int32)
    for mel_Index in range(mel_length):
        for token_Index in range(max(0, token_length + mel_Index - mel_length), min(token_length, mel_Index + 1)):
            if mel_Index == token_Index:
                current_Q = -1e+7
            else:
                current_Q = x[token_Index, mel_Index - 1]   # Stayed current token
            if token_Index == 0:
                if mel_Index == 0:
                    prev_Q = 0.0
                else:
                    prev_Q = -1e+7
            else:
                prev_Q = x[token_Index - 1, mel_Index - 1]  # Moved to next token
            x[token_Index, mel_Index] = max(current_Q, prev_Q) + x[token_Index, mel_Index]

    token_Index = token_length - 1
    for mel_Index in range(mel_length - 1, -1, -1):
        path[token_Index, mel_Index] = 1
        if token_Index == mel_Index or x[token_Index, mel_Index - 1] < x[token_Index - 1, mel_Index - 1]:
            token_Index = max(0, token_Index - 1)

    return path

def maximum_path(log_p, mask):
        '''
        x: [Batch, Token_t, Mel_t]
        mask: [Batch, Token_t, Mel_t]
        '''
        log_p *= mask
        device, dtype = log_p.device, log_p.dtype
        log_p = log_p.data.cpu().numpy().astype(np.float32)
        mask = mask.data.cpu().numpy()

        token_Lengths = np.sum(mask, axis= 1)[:, 0].astype(np.int32)   # [Batch]
        mel_Lengths = np.sum(mask, axis= 2)[:, 0].astype(np.int32)   # [Batch]

        paths = calc_paths(log_p, token_Lengths, mel_Lengths)

        return torch.from_numpy(paths).to(device= device, dtype= dtype)

class GradTTS(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats #80
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if self.n_spks > 1:
            self.spk_emb = torch.nn.Embedding(self.n_spks, self.spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        # x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        # x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths]) #텐서를 그냥 gpu,cpu로 옮기는 역할

        if self.n_spks > 1:
            # Get speaker embedding
            spk = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const #이게 loss_enc

            attn = maximum_path(log_prior, attn_mask.squeeze(1)) #(16,265,812)
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths) #logw : 예측된 값, logw_: 실제값(MAS로 구한 값)
        #그냥 오차제곱 loss
        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)): #y:(16,80,812)
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut #attn:(16,265,812)->(16,265,172)
            y = y_cut #y:(16,80,812)->(16,80,172)
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        # attn.squeeze(1).transpose(1, 2) : (16,172,265), mu_x.transpose(1, 2): (16,265,80)
        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss


class MyGradTTS(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(MyGradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.phonemes_encoder=PhonemesEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.notes_encoder=NotesEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)
        
        self.combining_layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=160, out_channels=80, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=80, out_channels=80, kernel_size=3, padding=1)
        )
    @torch.no_grad()
    def forward(self, notes, phonemes, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        # notes, phonemes = self.relocate_input([notes, phonemes])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x_phonemes = self.phonemes_encoder(phonemes,spk) #(1,80,192)
        mu_x_notes = self.notes_encoder(notes,spk)#(1,80,192)
        
        combined = torch.cat((mu_x_phonemes, mu_x_notes), dim=1)  # (batch, 160, 172)
        mu_y= self.combining_layers(combined)
        encoder_outputs=mu_y
        y_mask=torch.ones(mu_y.shape[0],1,mu_y.shape[2],device=mu_y.device).to(mu_y.dtype)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        
        return encoder_outputs, decoder_outputs
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def compute_loss(self, notes, phonemes, y, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        # notes, phonemes, y = self.relocate_input([notes, phonemes, y]) #텐서를 그냥 gpu,cpu로 옮기는 역할

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x_phonemes = self.phonemes_encoder(phonemes,spk) #(1,80,192)
        mu_x_notes = self.notes_encoder(notes,spk)#(1,80,192)
        combined = torch.cat((mu_x_phonemes, mu_x_notes), dim=1)  # (batch, 160, 172)
        mu_y= self.combining_layers(combined)
        
        y=y.transpose(1,2)
        y_mask=torch.ones(y.shape[0],1,y.shape[2],device=y.device).to(y.dtype)
        
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return  prior_loss,diff_loss
