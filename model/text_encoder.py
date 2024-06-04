""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch

from model.base import BaseModule
from model.utils import sequence_mask, convert_pad_shape


class LayerNorm(BaseModule):
    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, 
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, 
                                                kernel_size, padding=kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels, 
                                                    kernel_size, padding=kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(BaseModule):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super(DurationPredictor, self).__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, 
                                      kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, 
                                      kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class MultiHeadAttention(BaseModule):
    def __init__(self, channels, out_channels, n_heads, window_size=None, 
                 heads_share=True, p_dropout=0.0, proximal_bias=False, 
                 proximal_init=False):
        super(MultiHeadAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(torch.randn(n_heads_rel, 
                             window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = torch.nn.Parameter(torch.randn(n_heads_rel, 
                             window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        
    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, 
                                                                    dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, 
                                                                value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = torch.nn.functional.pad(
                            relative_embeddings, convert_pad_shape([[0, 0], 
                            [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                   slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0,0],[0,0],[0,length-1]]))
        x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
        x_flat = x.view([batch, heads, length**2 + length*(length - 1)])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(BaseModule):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, 
                 p_dropout=0.0):
        super(FFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, 
                                      padding=kernel_size//2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, 
                                      padding=kernel_size//2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(BaseModule):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, 
                 kernel_size=1, p_dropout=0.0, window_size=None, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels # 각 레이어의 히든 상태 채널 수.
        self.filter_channels = filter_channels #피드포워드 네트워크의 필터 채널 수.
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels,
                                    n_heads, window_size=window_size, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels,
                                       filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoder(BaseModule):
    def __init__(self, n_vocab, n_feats, n_channels, filter_channels, 
                 filter_channels_dp, n_heads, n_layers, kernel_size, 
                 p_dropout, window_size=None, spk_emb_dim=64, n_spks=1):
        super(TextEncoder, self).__init__()
        self.n_vocab = n_vocab #149
        self.n_feats = n_feats #80
        self.n_channels = n_channels # 임베딩 벡터의 차원 수,192
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp #256
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.emb = torch.nn.Embedding(n_vocab, n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        self.prenet = ConvReluNorm(n_channels, n_channels, n_channels, 
                                   kernel_size=5, n_layers=3, p_dropout=0.5)

        self.encoder = Encoder(n_channels + (spk_emb_dim if n_spks > 1 else 0), filter_channels, n_heads, n_layers, 
                               kernel_size, p_dropout, window_size=window_size)

        self.proj_m = torch.nn.Conv1d(n_channels + (spk_emb_dim if n_spks > 1 else 0), n_feats, 1) #스펙트로그램을 생성하기 위한 컨볼루션 레이어
        self.proj_w = DurationPredictor(n_channels + (spk_emb_dim if n_spks > 1 else 0), filter_channels_dp, 
                                        kernel_size, p_dropout)

    def forward(self, x, x_lengths, spk=None):
        x = self.emb(x) * math.sqrt(self.n_channels) #텍스트 입력을 임베딩 벡터로 변환하고, 스케일링. x : (16,265)->(16,265:순서,192)
        x = torch.transpose(x, 1, -1) #(batch_size=16, embedding_dim=192, sequence_length=265)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype) #(16,1,265)

        x = self.prenet(x, x_mask)#(batch_size=16, embedding_dim=192, sequence_length=265)
        if self.n_spks > 1:
            x = torch.cat([x, spk.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)#(16,192,265)
        mu = self.proj_m(x) * x_mask #mu예측#(16,80,265)

        x_dp = torch.detach(x) 
        logw = self.proj_w(x_dp, x_mask) #로그-길이를 예측

        return mu, logw, x_mask


class PhonemesEncoder(BaseModule):
    def __init__(self, n_vocab, n_feats, n_channels, filter_channels, 
                 filter_channels_dp, n_heads, n_layers, kernel_size, 
                 p_dropout, window_size=None, spk_emb_dim=64, n_spks=1):
        super(PhonemesEncoder, self).__init__()
        self.n_vocab = n_vocab=47 #149
        self.n_feats = n_feats=80 #80
        self.n_channels = n_channels = 256 # 임베딩 벡터의 차원 수,192
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp #256
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.emb = torch.nn.Embedding(n_vocab, n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        self.prenet = ConvReluNorm(n_channels, n_channels, n_channels, 
                                   kernel_size=5, n_layers=3, p_dropout=0.5)

        self.encoder = Encoder(n_channels + (spk_emb_dim if n_spks > 1 else 0), filter_channels, n_heads, n_layers, 
                               kernel_size, p_dropout, window_size=window_size)

        self.proj_m = torch.nn.Conv1d(n_channels + (spk_emb_dim if n_spks > 1 else 0), n_feats, 1) #스펙트로그램을 생성하기 위한 컨볼루션 레이어
        

    def forward(self, x, spk=None):
        x = self.emb(x) * math.sqrt(self.n_channels) #텍스트 입력을 임베딩 벡터로 변환하고, 스케일링. x : (16,265)->(16,265:순서,192)
        x = torch.transpose(x, 1, -1) #(batch_size=16, embedding_dim=192, sequence_length=265)
        x_mask=torch.ones((x.shape[0], 1, x.shape[2]),device=x.device).to(x.dtype)
        x = self.prenet(x, x_mask)#(batch_size=16, embedding_dim=192, sequence_length=265), (1,256,192)
        if self.n_spks > 1:
            x = torch.cat([x, spk.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)#(16,192,265),(1,256,192)
        mu = self.proj_m(x) * x_mask #mu예측 #(16,80,265),(1,80,192)
        return mu

class NotesEncoder(BaseModule):
    def __init__(self, n_vocab, n_feats, n_channels, filter_channels, 
                 filter_channels_dp, n_heads, n_layers, kernel_size, 
                 p_dropout, window_size=None, spk_emb_dim=64, n_spks=1):
        super(NotesEncoder, self).__init__()
        self.n_vocab = n_vocab=47#32
        self.n_feats = n_feats=80 #80
        self.n_channels = n_channels = 256 # 임베딩 벡터의 차원 수,192
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp #256
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.emb = torch.nn.Embedding(n_vocab, n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        self.prenet = ConvReluNorm(n_channels, n_channels, n_channels, 
                                   kernel_size=5, n_layers=3, p_dropout=0.5)

        self.encoder = Encoder(n_channels + (spk_emb_dim if n_spks > 1 else 0), filter_channels, n_heads, n_layers, 
                               kernel_size, p_dropout, window_size=window_size)

        self.proj_m = torch.nn.Conv1d(n_channels + (spk_emb_dim if n_spks > 1 else 0), n_feats, 1) #스펙트로그램을 생성하기 위한 컨볼루션 레이어
        

    def forward(self, x, spk=None):
        x = self.emb(x) * math.sqrt(self.n_channels) #텍스트 입력을 임베딩 벡터로 변환하고, 스케일링. x : (16,265)->(16,265:순서,192)
        x = torch.transpose(x, 1, -1) #(batch_size=16, embedding_dim=192, sequence_length=265)
        x_mask=torch.ones((x.shape[0], 1, x.shape[2]),device=x.device).to(x.dtype)
        x = self.prenet(x, x_mask)#(batch_size=16, embedding_dim=192, sequence_length=265), (1,256,192)
        if self.n_spks > 1:
            x = torch.cat([x, spk.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)#(16,192,265),(1,256,192)
        mu = self.proj_m(x) * x_mask #mu예측 #(16,80,265),(1,80,192)
        return mu
