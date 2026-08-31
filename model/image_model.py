# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F



from .layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2, SFMA
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import math

g_ch_src = 8 * 8
g_ch_enc_dec = 128
# g_ch_enc_dec = 368


class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

        """ 
        self.encoder_sfmas = nn.Sequential(
            SFMA(g_ch_enc_dec),
            SFMA(g_ch_enc_dec),
            SFMA(g_ch_enc_dec),
            SFMA(g_ch_enc_dec)
        )
        """
        

    def forward(self, x, quant_step):

        out = F.pixel_unshuffle(x, 8)

        return self.forward_torch(out, quant_step)


    def forward_torch(self, out, quant_step):
        out = self.enc_1(out)
        out = out * quant_step
        # out = self.encoder_sfmas[0](out)

        out1 = self.enc_2[0](out)
        out1 = self.enc_2[1](out1)
        # out2 = self.encoder_sfmas[1](out1)

        out2 = self.enc_2[2](out1)
        out2 = self.enc_2[3](out2)
        # out3 = self.encoder_sfmas[2](out2)

        out3 = self.enc_2[4](out2)
        out3 = self.enc_2[5](out3)
        # out4 = self.encoder_sfmas[3](out3)

        out5 = self.enc_2[6](out3)

        return out5

    def forward_cuda(self, out, quant_step):
        out = self.enc_1(out, quant_step=quant_step)
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src)

        """
        self.decoder_sfmas = nn.Sequential(
            SFMA(g_ch_enc_dec),
            SFMA(g_ch_enc_dec),
            SFMA(g_ch_enc_dec),
            SFMA(g_ch_enc_dec)
        )
        """
        

    def forward(self, x, quant_step):

        return self.forward_torch(x, quant_step)


    def forward_torch(self, x, quant_step):
        # out = self.dec_1(x)
        out1 = self.dec_1[0](x)

        # out1 = self.decoder_sfmas[0](out1)
        out2 = self.dec_1[1](out1)
        out2 = self.dec_1[2](out2)
        out2 = self.dec_1[3](out2)
        # out2 = self.decoder_sfmas[0](out2)

        out3 = self.dec_1[4](out2)
        out3 = self.dec_1[5](out3)
        out3 = self.dec_1[6](out3)
        # out3 = self.decoder_sfmas[1](out3)

        out4 = self.dec_1[7](out3)
        out4 = self.dec_1[8](out4)
        out4 = self.dec_1[9](out4)
        # out4 = self.decoder_sfmas[2](out4)

        out6 = self.dec_1[10](out4)
        out6 = self.dec_1[11](out6)
        out6 = self.dec_1[12](out6)
        # out6 = self.decoder_sfmas[3](out6)

        out7 = out6 * quant_step
        out = self.dec_2(out7)
        out = F.pixel_shuffle(out, 8)
        
        return out

    # def forward_cuda(self, x, quant_step):
    #     out = self.dec_1[0](x)
    #     out = self.dec_1[1](out)
    #     out = self.dec_1[2](out)
    #     out = self.dec_1[3](out)
    #     out = self.dec_1[4](out)
    #     out = self.dec_1[5](out)
    #     out = self.dec_1[6](out)
    #     out = self.dec_1[7](out)
    #     out = self.dec_1[8](out)
    #     out = self.dec_1[9](out)
    #     out = self.dec_1[10](out)
    #     out = self.dec_1[11](out)
    #     out = self.dec_1[12](out, quant_step=quant_step)
    #     out = self.dec_2(out)
    #     out = F.pixel_shuffle(out, 8)
    #     return out

def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    return padding_right, padding_bottom
def pad_for_y(y):
    _, _, H, W = y.size()
    padding_r, padding_b = get_padding_size(H, W, 4)
    y_pad = F.pad(y, (0, padding_r, 0, padding_b), mode="replicate")
    
    return y_pad

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class DMCI(nn.Module):
    def __init__(self, N=256, z_channel=128): #(self, N=128, z_channel=64)
        super().__init__()

        self.enc = IntraEncoder(N)
        
        self.entropy_bottleneck = EntropyBottleneck(z_channel)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N, 1),
            # nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        self.dec = IntraDecoder(N)

        self.q_scale_enc = nn.Parameter(torch.ones((64, g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((64, g_ch_enc_dec, 1, 1)))

    @torch.no_grad()
    def update(self, force=False,
               scale_min=0.11, scale_max=256.0, levels=64):
        """
        初始化/刷新 熵模型所需的 CDF 与 scale_table。
        - EntropyBottleneck.update(force): 量化参数->CDF表
        - GaussianConditional.update(scale_table): 注册离散scale表并生成CDF
        """
        # 1) z 的 EB CDF
        self.entropy_bottleneck.update(force=force)

        # 2) y 的 GC 需要一张离散的 scale_table
        #   （对数均匀取样是 CompressAI 常用做法，范围与levels可按需要调整）
        scale_table = torch.exp(
            torch.linspace(math.log(scale_min), math.log(scale_max), levels)
        ).to(next(self.parameters()).device)

        # 新版 CompressAI: update(scale_table)
        # 旧版也通常兼容，无参时需先 set scale_table，再 update()
        try:
            self.gaussian_conditional.update(scale_table)
        except TypeError:
            # 如果你的版本需要先赋值属性再update：
            self.gaussian_conditional.scale_table = scale_table
            self.gaussian_conditional.update()

        return True

    def forward(self, x, qp):

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)

        hyper_inp = pad_for_y(y)
        z = self.hyper_enc(hyper_inp)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)

        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()

        y_hat, y_likelihoods = self.gaussian_conditional(y, params)
        
        x_hat = self.dec(y_hat, curr_q_dec)#.clamp_(0, 1)
        return  x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods
    
    def forward_enc(self, x, qp):
        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        hyper_inp = pad_for_y(y)
        z = self.hyper_enc(hyper_inp)
        return hyper_inp, z
    
    def forward_dec(self, y, z, qp):
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)

        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()

        y_hat, y_likelihoods = self.gaussian_conditional(y, params)
        x_hat = self.dec(y_hat, curr_q_dec)#.clamp_(0, 1)
        return  x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods

    def compress(self, x, qp):

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]
        
        y = self.enc(x, curr_q_enc)
        hyper_inp = pad_for_y(y)
        z = self.hyper_enc(hyper_inp)

        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        indexes = self.gaussian_conditional.build_indexes(params)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return y_strings, z_strings, z.size()[-2:], y.shape

    def decompress(self, y_strings, z_strings, shapez, shapey, qp):

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        z_hat = self.entropy_bottleneck.decompress(z_strings, shapez)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = shapey
        params = params[:, :, :yH, :yW].contiguous()
        
        indexes = self.gaussian_conditional.build_indexes(params)
        
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)
        
        x_hat = self.dec(y_hat, curr_q_dec)#.clamp_(0, 1)
        
        return x_hat
