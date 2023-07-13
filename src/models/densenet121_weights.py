import torch
import torchvision.models as models


def download_densenet121_weights():
    model = models.densenet121(pretrained=True)
    torch.save(model.state_dict(), "densenet121_pretrained_weights.pth")


if __name__ == "__main__":
    download_densenet121_weights()
