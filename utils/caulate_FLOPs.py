from nets.MRF3Net import MRF3Net
import torch
from thop import profile
# Params: 0.162528M
# FLOPs: 1.410618GFLOPs
def show_model_params(num_classes, Train):
    input_img = torch.rand(1, 1, 256, 256).cuda()
    net = MRF3Net(num_classes=num_classes, Train=Train).cuda()
    flops, params = profile(net, inputs=(input_img,))
    print('\n')
    print('Model Params: %2fM || ' % (params / 1e6), end='')
    print('Model FLOPs: %2fGFLOPs' % (flops / 1e9))\

show_model_params(2, False)