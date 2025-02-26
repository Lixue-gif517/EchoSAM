# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from functools import partial
from .modeling.Echosam import echosam
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer,kpdecoder
from .modeling.kpdecoder import KeyPointDetector
from torch.nn import functional as F


# def build_samus_vit_h(args, checkpoint=None):
#     return _build_samus(
#         args,
#         encoder_embed_dim=1280,
#         encoder_depth=32,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[7, 15, 23, 31],
#         checkpoint=checkpoint,
#     )
#
#
# build_samus = build_samus_vit_h
#
#
# def build_samus_vit_l(args, checkpoint=None):
#     return _build_samus(
#         args,
#         encoder_embed_dim=1024,
#         encoder_depth=24,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[5, 11, 17, 23],
#         checkpoint=checkpoint,
#     )


def build_echosam_kp_vit_b(args, checkpoint=None):
    return _build_echosam_kp(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )



echosam_kp_model_registry = {
    "vit_b": build_echosam_kp_vit_b,
}


def _build_echosam_kp(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.encoder_input_size
    patch_size = image_size//32
    image_embedding_size = image_size // patch_size
    Echosam = echosam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size= patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        KeyPointDecoder=KeyPointDetector(),
    )
    Echosam.eval()
    # model = nn.DataParallel(model).cuda()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            Echosam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from2(Echosam, state_dict, image_size, patch_size)
            Echosam.load_state_dict(new_state_dict)
    return Echosam



def load_from(samus, sam_dict, image_size, patch_size):
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    token_size = int(image_size//patch_size)
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict


def load_from2(samus, sam_dict, image_size, patch_size): # load the positional embedding
    samus_dict = samus.state_dict()  #
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict and "mask_decoder.output_hypernetworks_mlps" not in k}  # 预训练的

    token_size = int(image_size//patch_size)
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    # for mlps in samus.mask_decoder.output_hypernetworks_mlps:
    #     for mlp in mlps:
    #         if isinstance(mlp,nn.Linear):
    #             nn.init.kaiming_uniform_(mlp.weight,a=0.0,mode="fan_in", nonlinearity="linear")
    #             nn.init.constant_(mlp.bias,0.0)
    samus_dict.update(dict_trained)
    return samus_dict
