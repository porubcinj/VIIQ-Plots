import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math
from modules.network import *
from utils import Config
from backbones.iresnet import iresnet50

class irisRecognition(object):
    def __init__(self, config: Config = Config()):
        self.cuda = config.CUDA
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")

        self.polar_height = config.POLAR_HEIGHT
        self.polar_width = config.POLAR_WIDTH

        self.nn_model_path = os.path.join("models", config.MODEL)

        self.NET_INPUT_SIZE = (320, 240)

        with torch.inference_mode():
            self.nn_model = iresnet50()
            self.nn_model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(768, 2048)
            )
            try:
                self.nn_model.load_state_dict(torch.load(self.nn_model_path, map_location=self.device), strict=False)
            except AssertionError:
                print("assertion error")
                self.nn_model.load_state_dict(torch.load(self.nn_model_path,
                    map_location = lambda storage, loc: storage))
            self.nn_model = self.nn_model.float().to(self.device)
            self.nn_model = self.nn_model.eval()

            self.input_transform_circ = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])
                    
        self.ISO_RES = (640,480)

    @torch.inference_mode()
    def extractVector(self, polar):
        im_polar = Image.fromarray(polar, "L")
        im_tensor = torch.tensor(np.array(im_polar).astype(np.float32)).unsqueeze(0)
        im_mean = im_tensor.mean()
        im_std = im_tensor.std()
        im_tensor = torch.clamp(torch.nan_to_num(((im_tensor - im_mean) / im_std), nan=4.0, posinf=4.0, neginf=-4.0), -4.0, 4.0).float()
        im_tensor = im_tensor.unsqueeze(0).repeat(1,3,1,1).to(self.device)
        vector = self.nn_model(im_tensor)
        return vector.cpu().numpy()[0]
        
    @torch.inference_mode()
    def matchVectors(self, vector1, vector2):
        dist = 0.0
        for val1, val2 in zip(vector1.tolist(), vector2.tolist()):
            dist += math.pow(val1 - val2, 2)
        return math.sqrt(dist)