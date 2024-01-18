from torch import nn
from models import resnet


# resnet
class Res_net(nn.Module):
    def __init__(self, device, num_classes):
        super(Res_net, self).__init__()
        self.device = device
        self.pretrained_model = resnet.resnet18(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 1, num_classes)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        res_logits = resnet_out
        return res_logits
