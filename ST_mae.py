# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import random
import logging
import os
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderViT(nn.Module):
    """
     ST_MAE with VisionTransformer backbone
    """

    def __init__(self, args, img_size=448, patch_size=16, in_chans=5,
                 embed_dim=1024,chan_embed_dim=768, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., chan_mask_num=2, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.args = args
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.spa_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.spa_patch_embed.num_patches

        self.chan_patch_embed = PatchEmbed(img_size, patch_size, in_chans - chan_mask_num, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks_Spa = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for i in range(depth)])
        self.blocks_Chan = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.chan_decoder_embed = nn.Linear(1280, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.chan_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks_Spa = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for i in range(decoder_depth)])
        self.decoder_blocks_Chan = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # if args.is_load_pretrain == 1:
        #     self._load_mae_all(args)

        self.fc = nn.Linear(embed_dim, chan_embed_dim, bias=True)

        self.un_Spa_decoder_embed = nn.Linear(embed_dim, 1280, bias=True)
        self.device = self._acquire_device()

    def print_log(self, message):
        print(message)
        logging.info(message)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            self.print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.print_log('Use CPU')
        return device

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.spa_patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.spa_patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.spa_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.spa_patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, n):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.spa_patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, n))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], n, h * p, h * p))
        return imgs

    def random_chan_mask(self, x, mask_num=2):
        """
        - images: Tensor of shape (B, 5, 448, 448)
        - num_to_remove: int, Number of pictures to be removed, Default 2
        - remaining_images: Tensor of shape (B, 3, 448, 448)
        - remove_indices: List of int, Index that is removed.
        """
        B, C, H, W = x.size()
        assert mask_num <= 5, "Cannot remove more images than are present"

        remove_indices = random.sample(range(5), mask_num)
        remove_indices = sorted(remove_indices)

        remaining_indices = [i for i in range(5) if i not in remove_indices]

        x_target = x[:, remove_indices, :, :]

        x = x[:, remaining_indices, :, :]  # b, 5-mask_num, h, w

        return x, remove_indices, x_target

    def random_masking(self, x, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_Spa_encoder(self, x, mask_ratio):
        # ------------------------------ space mask ---------------------------------------
        # embed patches
        x = self.spa_patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks_Spa:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_Spa_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks_Spa:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_Spa_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_Chan_encoder(self, x, mask_chan_num):
        B, C, H, W = x.size()
        # ------------------------------ chan mask -------------------------------------
        x_c, remove_indices, x_target = self.random_chan_mask(x, mask_num=mask_chan_num)  # b, 5-mask_num, h, w

        # x_c = x_c[:, 2:, :, :]  # train

        x_c = self.chan_patch_embed(x_c)  # b, h*w // patchsize **2, patchsize**2 * （5- mask_num）


        # apply Transformer blocks
        for blk in self.blocks_Chan:
            x_c = blk(x_c)
        x_c = self.norm(x_c)

        x_c = self.fc(x_c)
        x_c = self.unpatchify(x_c, 3)

        mask_c = torch.ones(B, mask_chan_num, H, W)
        mask_c = mask_c.to(self.device)

        x = torch.cat((x_c, mask_c), dim=1)

        return x, remove_indices, x_target

    def forward_Chan_decoder(self, x):
        x = self.patchify(x)
        # embed tokens
        x = self.chan_decoder_embed(x)

        # add pos embed
        x = x + self.chan_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks_Chan:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        x = self.unpatchify(x,5)

        return x

    def forward_Chan_loss(self, x, remove_indices, x_target, mask_chan_num):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target =x_target
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        pred = x[:, remove_indices, :, :]

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss.sum() / mask_chan_num  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, mask_chan_num=2):

        # ------------------pretrain-------------------------
        latent, mask, ids_restore = self.forward_Spa_encoder(imgs, mask_ratio)
        pred = self.forward_Spa_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss_Spa = self.forward_Spa_loss(imgs, pred, mask)

        x_c, remove_indices, target = self.forward_Chan_encoder(imgs, mask_chan_num)
        x = self.forward_Chan_decoder(x_c)
        loss_Chan = self.forward_Chan_loss(x, remove_indices, target, mask_chan_num)
        return loss_Spa, loss_Chan, pred, x, mask
        # ------------------train----------------------------
        # latent, mask, ids_restore = self.forward_Spa_encoder(imgs, mask_ratio)
        # # remove cls_token
        # latent = latent[:, 1:, :]
        # latent = self.un_Spa_decoder_embed(latent)

        # x_s = self.unpatchify(latent, 5)
        # x_c, remove_indices, target = self.forward_Chan_encoder(imgs, mask_chan_num)
        # return x_s, x_c


