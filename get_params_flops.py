# from nets.efficientdet import EfficientDetBackbone
#
# # 可替换为自己的模型及输入
# # 创建模型
# model = EfficientDetBackbone(20,0)
#
# total_num = sum(p.numel() for p in model.parameters())
# trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Total %s',total_num)
# print('Trainable %s',trainable_num)
# print("Number of parameter: %.3fM" % (total_num / 1e6))


import torch
import numpy as np
from torch.autograd import Variable
from nets.efficientdet import EfficientDetBackbone

def count_params(model, input_size=512):
    # param_sum = 0
    with open('models.txt', 'w') as fm:
        fm.write(str(model))

    # 计算模型的计算量
    calc_flops(model, input_size)

    # 计算模型的参数总量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('The network has {} params.'.format(params))


# 计算模型的计算量
def calc_flops(model, input_size):
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    # if '0.4.' in torch.__version__:
    #     if assets.USE_GPU:
    #         input = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
    #     else:
    #         input = torch.FloatTensor(torch.rand(2, 3, input_size, input_size))
    # else:
    #     input = Variable(torch.rand(2, 3, input_size, input_size), requires_grad=True)
    input = Variable(torch.rand(2, 3, input_size, input_size), requires_grad=True)
    _ = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.3fM' % (total_flops / 1e6 / 2))


if __name__ == "__main__":
    # 只需要更改版本号即可
    phi = 2
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
    model = EfficientDetBackbone(20,phi)
    count_params(model, input_sizes[phi])