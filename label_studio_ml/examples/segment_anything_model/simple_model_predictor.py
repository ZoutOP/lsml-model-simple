import logging
import cv2

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch
import os

from typing import Optional, Tuple

from label_studio_ml.utils import InMemoryLRUDictCache
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

from segment_anything.utils.transforms import ResizeLongestSide


logger = logging.getLogger(__name__)

LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")


class ModelSimple(nn.Module):
    """
    Wrapper for the sam model to to fine-tune the model on a new dataset

    ...
    Attributes:
    -----------
    freeze_encoder (bool): freeze the encoder weights
    freeze_decoder (bool): freeze the decoder weights
    freeze_prompt_encoder (bool): freeze the prompt encoder weights
    transform (ResizeLongestSide): resize the images to the model input size

    Methods:
    --------
    setup(): load the model and freeze the weights
    forward(images, points): forward pass of the model, returns the masks and iou_predictions
    """

    def __init__(self, freeze_encoder=True, freeze_decoder=False, freeze_prompt_encoder=True, use_input: bool = True, format_predictions: bool = False):
        super().__init__()
        self.model = None

        self.use_input = use_input
        self.format_predictions = format_predictions
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        # we need this to make the input image size compatible with the model
        self.transfrom = ResizeLongestSide(1024)  # default size.
        # Image cache when prediction on
        self.image_cache = InMemoryLRUDictCache(1)  # Copied from same_predictor.

    def __call__(self, model) -> "ModelSimple":
        """ Wrap the call (model) to initiate the model simple. This way we can initialise the model simple with parameters first. """
        self.setup(model)
        return self  # Return self.        

    def setup(self, model):
        """ Setup model."""
        self.model = model
        # to speed up training time, we normally freeze the encoder and decoder
        if self.freeze_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        self.transfrom = ResizeLongestSide(self.model.image_encoder.img_size)

    def get_image(self, image_path, task: Optional[dict] = None, image_format: str = "RGB") -> torch.Tensor:
        """ Get and return image from cache or image path. """
        image = self.image_cache.get(image_path)
        if image is not None:
            return image
        logger.debug(f'Payload not found for {image_path} in `IN_MEM_CACHE`: calculating from scratch')
        image_path = get_local_path(
            image_path,
            access_token=LABEL_STUDIO_ACCESS_TOKEN,
            hostname=LABEL_STUDIO_HOST,
            task_id=task.get('id')
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Parse to numpy
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]
        
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        assert (
            len(input_image_torch.shape) == 4
            and input_image_torch.shape[1] == 3
            and max(*input_image_torch.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

        self.image_cache.put(image_path, input_image_torch)
        return input_image_torch    

    @torch.no_grad()
    def forward(self, images, 
                point_coords: Optional[torch.Tensor],
                point_labels: Optional[torch.Tensor],
                boxes: Optional[torch.Tensor] = None,
                mask_input: Optional[torch.Tensor] = None):
        _, _, H, W = images.shape # batch, channel, height, width
        
        image_embeddings = self.model.image_encoder(images) # shape: (1, 256, 64, 64)
        # get prompt embeddings without acutally any prompts (uninformative)

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None if not self.use_input else points,
            boxes=None if not self.use_input else boxes,
            masks=None if not self.use_input else mask_input,
        )

        # get low resolution masks and iou predictions
        # mulitmask_output=False means that we only get one mask per image,
        # otherwise we would get three masks per image
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings, # sparse_embeddings shape: (1, 0, 256)
            dense_prompt_embeddings=dense_embeddings, # dense_embeddings shape: (1, 256, 256)
            multimask_output=False,
        )
        # postprocess the masks to get the final masks and resize them to the original image size
        masks = F.interpolate(
            low_res_masks, # shape: (1, 1, 256, 256)
            (H, W),
            mode="bilinear",
            align_corners=False,
        )
        # shape masks after interpolate: torch.Size([1, 1, 1024, 1024])
        return masks, iou_predictions
