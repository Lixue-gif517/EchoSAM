# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import shape
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
from models.segment_anything_echosam.modeling.bertwarper import BertModelWarper,generate_masks_with_special_tokens_and_transfer_map
# text_ = ["提取出","分割出","提取并分割出","帮我划分一下"]
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .kpdecoder import KeyPointDetector
from .text_encoder import FeatureEnhancer
from einops import rearrange


class echosam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        KeyPointDecoder:KeyPointDetector,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.KeyPointDecoder = KeyPointDecoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        text_encoder_type = "./checkpoints/bert-base-uncased"

        # bert文本编码器
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
        self.bert = BertModel.from_pretrained(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        self.max_text_len = 256
        self.sub_sentence_present = True
        self.feat_map = nn.Linear(self.bert.config.hidden_size, 256, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)


        for param in self.prompt_encoder.parameters():
          param.requires_grad = False
        for param in self.mask_decoder.parameters():
          param.requires_grad = True
        for n, value in self.image_encoder.named_parameters():
          if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
            value.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward_sam(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def forward(
        self, 
        imgs: torch.Tensor,
        targets: List = None,
        bbox: torch.Tensor = None,  # b 4
        pt: Tuple[torch.Tensor, torch.Tensor] = None,  # [b n 2, b n]
    ) -> torch.Tensor:
        if targets != None:
            imge= self.image_encoder(imgs)  # 待确定输出的shape

            captions = targets
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenized = self.tokenizer(captions, padding="longest",
                                       return_tensors="pt", truncation=True).to(device)
            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )

            if text_self_attention_masks.shape[1] > self.max_text_len:
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

            # extract text embeddings
            if self.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                tokenized_for_encoder = tokenized

            bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768
            encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            if encoded_text.shape[1] > self.max_text_len:
                encoded_text = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len]
            text_dict = {
                "encoded_text": encoded_text,  # bs, 195, d_model
                "text_token_mask": text_token_mask,  # bs, 195
                "position_ids": position_ids,  # bs, 195
                "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
            }  # 我想要的text embedding
            low_res_masks, _ = self.mask_decoder( # low_res_mask b 1 128 128
                        image_embeddings = imge,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        # sparse_prompt_embeddings=se,
                        text_dict = text_dict,
                        multimask_output=False,
                        )
            masks = F.interpolate(low_res_masks, (256, 256), mode="bilinear", align_corners=False)
            keypoints = self.KeyPointDecoder(imge,masks)

            outputs = {"low_res_logits": low_res_masks, "masks": masks,"keypoints":keypoints}
        else:
            imge = self.image_encoder(imgs)
            if len(pt[0].shape) == 3:
                se, de = self.prompt_encoder(  # se b 2 256, de b 256 32 32
                    points=pt,
                    boxes=None,
                    masks=None,
                )
                low_res_masks, _ = self.mask_decoder(  # low_res_mask b 1 128 128
                    image_embeddings=imge,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    text_dict=None,
                    multimask_output=False,
                )
                masks = F.interpolate(low_res_masks, (256, 256), mode="bilinear", align_corners=False)
                keypoints = self.KeyPointDecoder(imge, masks)
                outputs = {"low_res_logits": low_res_masks, "masks": masks,"keypoints":keypoints}
                return outputs
            else:
                low_res_masks, masks = [], []
                # for i in range(pt[0].shape[1]):
                #     pti = (pt[0][:, i, :, :], pt[1][:, i, :])
                #     sei, dei = self.prompt_encoder(  # se b 2 256, de b 256 32 32
                #         points=pti,
                #         boxes=None,
                #         masks=None,
                #     )
                sei, dei = self.prompt_encoder(  # se b 2 256, de b 256 32 32
                    points=None,
                    boxes=pt,
                    masks=None,
                )
                low_res_masksi, _ = self.mask_decoder(  # low_res_mask b 1 128 128
                    image_embeddings=imge,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sei,
                    dense_prompt_embeddings=dei,
                    multimask_output=False,
                )
                masksi = F.interpolate(low_res_masksi, (256, 256), mode="bilinear", align_corners=False)
                low_res_masks.append(low_res_masksi)
                masks.append(masksi)
                low_res_masks = torch.stack(low_res_masks, dim=1)
                masks = torch.stack(masks, dim=1)  # b c 1 255 255
                masks = masks.reshape(masks.shape[0], -1, masks.shape[3], masks.shape[4])
                low_res_masks = low_res_masks.reshape(low_res_masks.shape[0], -1, low_res_masks.shape[3],
                                                      low_res_masks.shape[4])
                keypoints = self.KeyPointDecoder(imge, masks)
                outputs = {"low_res_logits": low_res_masks, "masks": masks, "keypoints": keypoints}
        return outputs
        # else:
        #   low_res_masks, masks = [], []
        #   for i in range(pt[0].shape[1]):
        #     pti = (pt[0][:, i, :, :], pt[1][:, i, :])
        #     sei, dei = self.prompt_encoder(            # se b 2 256, de b 256 32 32
        #                 points=None,
        #                 boxes=bbox,
        #                 masks=None,
        #             )
        #     low_res_masksi, _ = self.mask_decoder( # low_res_mask b 1 128 128
        #             image_embeddings=imge,
        #             image_pe=self.prompt_encoder.get_dense_pe(),
        #             sparse_prompt_embeddings=sei,
        #             dense_prompt_embeddings=dei,
        #             multimask_output=False,
        #             )
        #     masksi = F.interpolate(low_res_masksi, (256, 256), mode="bilinear", align_corners=False)
        #     low_res_masks.append(low_res_masksi)
        #     masks.append(masksi)
        #   low_res_masks = torch.stack(low_res_masks, dim=1)
        #   masks = torch.stack(masks, dim=1) # b c 1 255 255
        #   masks = masks.reshape(masks.shape[0], -1, masks.shape[3], masks.shape[4])
        #   low_res_masks = low_res_masks.reshape(low_res_masks.shape[0], -1, low_res_masks.shape[3], low_res_masks.shape[4])
        #   outputs = {"low_res_logits": low_res_masks, "masks": masks}
        #   return outputs



    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
