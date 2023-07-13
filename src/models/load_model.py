import torch
import torch.nn as nn
from torchvision import models
import torchvision.models.detection as detection

from src.utils import args


# Model Selection

def load_model(model_name=args.model):
    if model_name == "resnet18":
        weights = detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
    elif model_name == "densenet121":
        # Load the model without pretrained weights
        model = models.densenet121(pretrained=False)
        num_features = model.classifier.in_features

        # Modify the classifier layer to match the size of the loaded checkpoint
        # Use the size from the checkpoint
        classifier = nn.Linear(num_features, 1000)
        model.classifier = classifier

        # Load the pretrained weights separately
        checkpoint = torch.load(
            "E:/thesis/thesis/src/models/densenet121_pretrained_weights.pth")

        # Check if the state dictionary key is present in the checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Adjust the classifier weights and biases
        state_dict['classifier.weight'] = state_dict['classifier.weight'][:num_features, :]
        state_dict['classifier.bias'] = state_dict['classifier.bias'][:num_features]

        # Load the adjusted state dictionary
        model.load_state_dict(state_dict)

        # Modify the classifier layer to match the desired number of output classes
        model.classifier = nn.Linear(num_features, 3)

    model = nn.DataParallel(model.to(args.device))
    return model


if __name__ == "__main__":
    pass
