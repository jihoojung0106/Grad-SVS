# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import params
from model.tts import  MyGradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols
from data_preprocess.core import make_loaders

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir = "logs/0604"
n_epochs = params.n_epochs
batch_size = params.batch_size #16
out_size = params.out_size #172
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols) #149
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


if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    accelerator = Accelerator()
    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    # print('Initializing data loaders...')
    # train_dataset = TextMelDataset(train_filelist_path, cmudict_path, add_blank,
    #                                n_fft, n_feats, sample_rate, hop_length,
    #                                win_length, f_min, f_max) #y : [80,759],x : [229]
    
    # batch_collate = TextMelBatchCollate()
    # loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
    #                     collate_fn=batch_collate, drop_last=True,
    #                     num_workers=4, shuffle=False)
    # test_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
    #                               n_fft, n_feats, sample_rate, hop_length,
    #                               win_length, f_min, f_max)
    loader=make_loaders("train")
    test_loader=make_loaders("test")
    it=iter(test_loader)
    test_batch=next(it)
    print('Initializing model...')
    model = MyGradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    
    
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    # model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    
    # for i, item in enumerate(test_batch):
    mel = test_batch[2][0].transpose(0,1)
    logger.add_image(f'image_0/ground_truth', plot_tensor(mel),global_step=0, dataformats='HWC')
    save_plot(mel, f'{log_dir}/original_{0}.png')
    
    print('Start training...')
    iteration = 0
    model.load(f"{log_dir}/grad_2_.pt")
    loader,  model, optimizer = accelerator.prepare(
        loader, model, optimizer
    )
    for epoch in range(1, n_epochs + 1):
        model.train()
        
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(loader)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                notes, phonemes, y = batch[0], batch[1], batch[2]
                
                prior_loss,diff_loss = model.module.compute_loss(notes, phonemes, y, out_size=out_size)
                loss = sum([prior_loss,diff_loss])
                accelerator.backward(loss)

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)
                
                iteration += 1

        log_msg = 'Epoch %d: ' % (epoch)
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % params.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            
            # for i, item in enumerate(test_batch):
                notes, phonemes, y = test_batch[0][0].unsqueeze(0).cuda(), test_batch[1][0].unsqueeze(0).cuda(), test_batch[2][0].unsqueeze(0).cuda()
                i=0
                # # x = item['x'].to(torch.long).unsqueeze(0).cuda()
                # x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec = model.module(notes, phonemes, n_timesteps=50)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                # logger.add_image(f'image_{i}/alignment',
                #                  plot_tensor(attn.squeeze().cpu()),
                #                  global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), f'{log_dir}/generated_enc_{i}.png')
                print(f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), f'{log_dir}/generated_dec_{i}.png')
                print(f'{log_dir}/generated_dec_{i}.png')
                # save_plot(attn.squeeze().cpu(), 
                #           f'{log_dir}/alignment_{i}.png')

        ckpt = model.module.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
