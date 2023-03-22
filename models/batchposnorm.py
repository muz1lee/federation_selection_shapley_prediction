import math

from torch import nn
import torch


def diff_penalty(model):
    penalty = []
    for m in model.modules():
        if isinstance(m, PosBatchNorm2d):
            penalty.append(m.diff)
            # penalty.append(torch.clamp(m.diff, max=0.2))
    # penalty = torch.tanh(torch.cat(penalty)) # sigmoid几乎没有什么更新，输出都是）0.7
    penalty = torch.cat(penalty)
    return penalty


class PosBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(PosBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.channel_num = num_features
        self.max_index = nn.Parameter(torch.Tensor([num_features - 1]), requires_grad=False)
        self.diff = nn.Parameter(torch.Tensor([1]).float(), requires_grad=False)
        self.register_parameter('diff', self.diff)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        indexes = torch.arange(self.channel_num).cuda()
        # slow_fast_indexes = torch.pow(indexes / self.channel_num, 1 / 3) * self.channel_num
        # attempt 2
        # func_sin = 1 / 2 * torch.sin(
        #     (torch.arange(self.channel_num).cuda() - self.diff) * math.pi / (2 * self.channel_num)) + 1


        # attempt3
        gap = 0.05
        mega = 10
        func_sin = 1 / 2 * (
                    torch.sin((indexes + 1 / 2 * self.channel_num) * math.pi / (self.channel_num)) - 1) * gap * mega + \
                   (1 - gap)
        func_one = torch.ones(self.channel_num).cuda()
        func_gap = (func_one - func_sin) * torch.clamp((1 - self.diff), min=0)
        pos_enc = func_sin + func_gap
        # pos_enc = torch.where(indexes < self.diff, func_sin, func_one)

        # old pos encoding: attempt1
        # pos_enc = torch.pow(0.999, torch.arange(self.channel_num)).cuda()
        # pos_enc[index + 1:] = pos_enc[index]

        input = (input - mean[None, :, None, None]) / (
            torch.sqrt(var[None, :, None, None] * pos_enc[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

# class PosBatchNorm(nn.Module):
#     def __init__(self, channel_num):
#         super(PosBatchNorm, self).__init__()
#         self.channel_num = channel_num
#
#         self.max_index = nn.Parameter(torch.Tensor([channel_num - 1]), requires_grad=False)
#         self.diff = nn.Parameter(torch.zeros([1]).float(), requires_grad=False)
#         # self.diff += self.channel_num - 1
#
#     def forward(self, x, bn):
#         index = torch.min(self.diff, self.max_index).floor().long()
#         pos_enc = torch.pow(0.999, torch.arange(self.channel_num)).cuda()
#         pos_enc[index + 1:] = pos_enc[index]
#         var_modified = (bn.running_var * pos_enc).view(1, -1, 1, 1)
#         bias = bn.bias.view(1, -1, 1, 1)
#         weight = bn.weight.view(1, -1, 1, 1)
#         var = bn.running_var.view(1, -1, 1, 1)
#         mid = (x - bias) / weight * torch.sqrt(var + bn.eps)
#         return mid / torch.sqrt(var_modified + bn.eps) * weight + bias
