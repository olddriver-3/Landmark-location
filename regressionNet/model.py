
import torch
import torchvision


def load_model(model='resnet50', n_landmarks=19):
    if model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, n_landmarks * 2)
        return model
    else:
        raise NotImplementedError


if __name__ == '__main__':
    model = load_model()
    print(model)
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)

