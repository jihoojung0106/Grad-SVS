# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model.tts import GradTTS,MyGradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from utils import plot_tensor, save_plot

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
import params
from data_preprocess.core import make_loaders
HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
batch_size = params.batch_size #16
out_size = params.out_size #172
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1#149
n_enc_channels = params.n_enc_channels #192
filter_channels = params.filter_channels #768
filter_channels_dp = params.filter_channels_dp #256
n_enc_layers = params.n_enc_layers #6
enc_kernel = params.enc_kernel #3
enc_dropout = params.enc_dropout #0.1
n_heads = params.n_heads #2
window_size = params.window_size #4

n_feats = params.n_feats #80
n_fft = params.n_fft #1024
sample_rate = params.sample_rate #22050
hop_length = params.hop_length #256
win_length = params.win_length #1024
f_min = params.f_min #0
f_max = params.f_max #8000

dec_dim = params.dec_dim #64
beta_min = params.beta_min #0.05
beta_max = params.beta_max #20.0
pe_scale = params.pe_scale  #1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--file', type=str, required=False, help='path to a file with texts to synthesize')
    # parser.add_argument('-c', '--checkpoint', type=str, required=False, help='path to a checkpoint of Grad-TTS')
    # parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    # parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    # args = parser.parse_args()
    
  
    
    print('Initializing Grad-TTS...')
    generator = MyGradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    
    # generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
    #                     params.n_enc_channels, params.filter_channels,
    #                     params.filter_channels_dp, params.n_heads, params.n_enc_layers,
    #                     params.enc_kernel, params.enc_dropout, params.window_size,
    #                     params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    
    generator.load_state_dict(torch.load("logs/0604/grad_2_.pt", map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    test_loader=make_loaders("test")
    it=iter(test_loader)
    test_batch=next(it)
    
    with torch.no_grad():
        cnt=0
        for j in range(len(test_loader)):
            test_batch=next(it)
            for i in range(test_batch[0].shape[0]):
                notes, phonemes, y = test_batch[0][i].unsqueeze(0).cuda(), test_batch[1][i].unsqueeze(0).cuda(), test_batch[2][i].cuda()
                y_enc, y_dec = generator(notes, phonemes, n_timesteps=50)
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                write(f'./out/{cnt}_generated.wav', 22050, audio)
                original_audio = (vocoder.forward(y.transpose(0,1).unsqueeze(0)).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                write(f'./out/{cnt}_original.wav', 22050, audio)
                save_plot(y_enc.squeeze().cpu(), f'./out/{cnt}_enc.png')
                save_plot(y_dec.squeeze().cpu(), f'./out/{cnt}_dec.png')
                save_plot(y.transpose(0,1).cpu(), f'./out/{cnt}_mel.png')
                print(f"./out/{cnt}_generated.wav")
                cnt+=1

    print('Done. Check out `out` folder for samples.')
