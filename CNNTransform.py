from torchvision import models
class ResNet18():
    def __init__(self, net):
        self.net = net
    def transform(self, x):
        v = self.net.conv1(x)
        v = self.net.bn1(v)
        v = self.net.relu(v)
        v = self.net.maxpool(v)
        v = self.net.layer1(v)
        v = self.net.layer2(v)
        v = self.net.layer3(v)
        #v = self.net.layer4(v)
        v = self.net.avgpool(v)
        return v.reshape(len(v),1024)

def resnet18():
    m = models.resnet18(pretrained=True)
    return ResNet18(m)
