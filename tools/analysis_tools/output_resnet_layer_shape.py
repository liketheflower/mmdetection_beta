from mmdet.models import ResNet
import torch
self = ResNet(depth=50)
self.eval()
inputs = torch.rand(1, 3, 800, 1333)
#inputs = torch.rand(1, 3, 224, 224)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))
