# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,  # transformer维度
        transformer: nn.Module,  # transformer结构
        num_multimask_outputs: int = 3,  # 预测mask的数量、消除歧义
        activation: Type[nn.Module] = nn.GELU,  # 上采样时使用的激活函数
        iou_head_depth: int = 3,  # 预测掩码质量的深度
        iou_head_hidden_dim: int = 256,  # 预测掩码质量的隐藏维度
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 上采样时，使用两次转置卷积
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 32, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 32),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 32, transformer_dim // 64, kernel_size=2, stride=2),
            activation(),

        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 64, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings=None,
        dense_prompt_embeddings=None,
        text_dict=None,
        multimask_output=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            text_dict=text_dict,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings=None,
        dense_prompt_embeddings=None,
            # 1 256 32 32
        text_dict=None, # b 256 32 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # 拼接iou和mask作为output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # 1 iou + 1 total mask + 3 sub mask = (5, 256)
        if sparse_prompt_embeddings is None:
            sparse_prompt_embeddings = torch.empty(1,0,256).to("cuda")
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # b 5 256
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # only cat with sparse_prompt # b 7 256 when sparse_prompt point is 2

        # Expand per-image data in batch direction to be per-mask
        if len(image_embeddings.shape) == 3:
            image_embeddings =  image_embeddings.unsqueeze(0)
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # b 256 32 32 when the decoder operated in bs=1
        else:
            src = image_embeddings
        # 融合过程，直接相加
        # ss = dense_prompt_embeddings["encoded_text"]
        # src = src + dense_prompt_embeddings["encoded_text"]
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # b 256 32 32
        b, c, h, w = src.shape

        # Run the transformer
        if text_dict != None:
            hs, src = self.transformer(src, pos_src, text_dict) # hs (b nt c), src (b N c)
            # iou_token_out = hs[:, 0, :]  # b c
            # mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # b 4 c
        else:
            hs, src = self.transformer(src, pos_src, text_dict=None,point_embedding=tokens)  # hs (b nt c), src (b N c)
        iou_token_out = hs[:, 0, :] # b c
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] # b 4 c

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w) 
        upscaled_embedding = self.output_upscaling(src) # b 32 4h 4w

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1) # b 4 32
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # b 4 4h 4w 

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out) # b 4

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
