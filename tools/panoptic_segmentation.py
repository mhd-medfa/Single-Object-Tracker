from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy
torch.set_grad_enabled(False)
import panopticapi
from panopticapi.utils import id2rgb, rgb2id

# These are the COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]



class PanopticSegmentation:
    def __init__(self) -> None:
        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model, self.postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
        self.model.eval()
        self.image = None
        self.normalized_image = None
        self.good_masks_ids = None
        self.postprocessed_result = None
        self.panoptic_seg = None
        self.panoptic_seg_id = None
        
        
    def _normalize_input(self, image=None):
        # mean-std normalize the input image (batch-size: 1)
        self.image = self.transform(image).unsqueeze(0)
        self.normalized_image = self.model(self.image)
    
    def _filter_good_masks(self):
        # compute the scores, excluding the "no-object" class (the last one)
        scores = self.normalized_image["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
        # threshold the confidence
        self.good_masks_ids = scores > 0.85
    def _postprocessing(self):
        # the post-processor expects as input the target size of the predictions (which we set here to the image size)
        self.postprocessed_result = self.postprocessor(self.normalized_image, torch.as_tensor(self.image.shape[-2:]).unsqueeze(0))[0]
        # The segmentation is stored in a special-format png
        self.panoptic_seg = Image.open(io.BytesIO(self.postprocessed_result['png_string']))
        self.panoptic_seg = numpy.array(self.panoptic_seg, dtype=numpy.uint8).copy()
        # We retrieve the ids corresponding to each mask
        self.panoptic_seg_id = rgb2id(self.panoptic_seg)

