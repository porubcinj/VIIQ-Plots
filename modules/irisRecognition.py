import os
import torch
from backbones.iresnet import iresnet50, iresnet100
from backbones.convnext import ArcFaceConvNeXt

class irisRecognition(object):
    def __init__(self, model: str, **kwargs):
        self.device = torch.device("mps")
        self.nn_model_path = os.path.join("models", model)

        with torch.inference_mode():
            if model.startswith("ResNet50"):
                self.nn_model = iresnet50(**kwargs)
            elif model.startswith("ResNet100"):
                self.nn_model = iresnet100(**kwargs)
            elif model.startswith("ConvNeXt"):
                convnext_model_name = '_'.join(model.split('_')[:2])
                self.nn_model = ArcFaceConvNeXt(convnext_model_name, **kwargs)
            else:
                raise ValueError(f"Invalid model name: {model}")

            state_dict = torch.load(self.nn_model_path, map_location=self.device)
            new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            self.nn_model.load_state_dict(new_state_dict, strict=True)
            self.nn_model = self.nn_model.float().to(self.device)
            self.nn_model = self.nn_model.eval()

    @torch.inference_mode()
    def extractVectors(self, im_tensors):
        return self.nn_model(im_tensors)
#
#    @torch.inference_mode()
#    def matchVectors(self, enroll_embeddings, search_embeddings):
#        distances: torch.Tensor = torch.linalg.vector_norm(enroll_embeddings - search_embeddings, dim=1)
#        return distances
#
    @torch.inference_mode()
    def matchVectorsArc(self, enroll_embeddings, search_embeddings):
        cosine_similarity = torch.nn.functional.cosine_similarity(enroll_embeddings, search_embeddings, dim=1)
        arc_distance = torch.acos(cosine_similarity)
        return arc_distance