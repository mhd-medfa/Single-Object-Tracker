from multiprocessing.dummy import Array
import cv2
import torch
import time
import numpy as np
import os


class MiDaS:
    def __init__(self, model_type="MiDaS_small") -> None:
        
        # Load a MiDas model for depth estimation
        model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        self.model_type = model_type  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Move model to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        # Load transforms to resize and normalize the image
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

    def estimate(self, img) -> np.ndarray:
        self.img = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply input transforms
        input_batch = self.transform(self.img).to(self.device)

        # Prediction and resize to original resolution
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                            ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

        # depth_map = (depth_map*255).astype(np.uint8)
        magma_depth_map = cv2.applyColorMap((depth_map*255).astype(np.uint8) , cv2.COLORMAP_MAGMA)
        
        return depth_map, magma_depth_map