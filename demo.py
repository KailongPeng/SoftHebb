"""
使用 SoftHebb 在 CIFAR10 上训练 ConvNet 的单文件脚本演示，SoftHebb 是一种无监督、高效和生物可信的学习算法
Demo single-file script to train a ConvNet on CIFAR10 using SoftHebb, an unsupervised, efficient and bio-plausible learning algorithm
"""
import math
import warnings

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR
import torchvision

import os
os.chdir('/gpfs/milgram/project/turk-browne/projects/SoftHebb')

class SoftHebbConv2d(nn.Module):
    # SoftHebbConv2d是一个自定义的卷积层，使用SoftHebb算法进行权重更新。
    # 初始化函数设置了卷积层的参数，包括输入通道数、输出通道数、卷积核大小等。
    # 权重初始化使用正态分布，并且有一个用于权重更新的参数t_invert。
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            t_invert: float = 12,
    ) -> None:
        """
        Simplified implementation of Conv2d learnt with SoftHebb; an unsupervised, efficient and bio-plausible
        learning algorithm.
        This simplified implementation omits certain configurable aspects, like using a bias, groups>1, etc. which can
        be found in the full implementation in hebbconv.py
        """
        # 初始化SoftHebbConv2d层的参数
        super(SoftHebbConv2d, self).__init__()
        assert groups == 1, "Simple implementation does not support groups > 1."
        # 确保简化实现不支持groups>1
        # 其他参数设置
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)
        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        self.t_invert = torch.tensor(t_invert)

    def forward(self, x):
        # forward函数定义了SoftHebbConv2d层的前向传播。
        # 在训练模式下，使用SoftHebb算法进行权重更新。
        # 首先计算后突激活，然后计算可塑性更新，并将更新后的梯度存储在self.weight.grad中。
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        # 执行卷积，得到加权输入u \in [B, OC, OH, OW]
        # perform conv, obtain weighted input u \in [B, OC, OH, OW]
        weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        if self.training:
            # 在训练模式下执行以下操作
            # ===== 找到后突激活 y = sign(u)*softmax(u, dim=C), s(u)=1 - 2*I[u==max(u,dim=C)] =====
            # 后突激活是通过softmax函数计算的，非获胜神经元接收到的激活为负的后突激活。
            # 找到每个输入元素和像素的获胜神经元
            # ===== find post-synaptic activations y = sign(u)*softmax(u, dim=C), s(u)=1 - 2*I[u==max(u,dim=C)] =====
            # Post-synaptic activation, for plastic update, is weighted input passed through a softmax.
            # Non-winning neurons (those not with the highest activation) receive the negated post-synaptic activation.
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            # 扁平化非竞争维度（B、OC、OH、OW）->（OC、B*OH*OW） # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            # 计算每个批次元素和像素的胜出神经元  # Compute the winner neuron for each batch element and pixel
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
            flat_softwta_activs = - flat_softwta_activs  # 将所有后突激活转换为反Hebbian # Turn all postsynaptic activations into anti-Hebbian
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # winning neuron for each pixel in each input
            competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
            # Turn winner neurons' activations back to hebbian
            flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]
            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # ===== compute plastic update Δw = y*(x - u*w) = y*x - (y*u)*w =======================================
            # Use Convolutions to apply the plastic update. Sweep over inputs with postynaptic activations.
            # Each weighting of an input pixel & an activation pixel updates the kernel element that connected them in
            # the forward pass.
            # ===== 计算可塑性更新 Δw = y*(x - u*w) = y*x - (y*u)*w =======================================
            # 使用卷积应用可塑性更新。在输入上遍历，使用后突激活的像素和输入像素的加权和来更新连接它们的卷积核元素。
            yx = F.conv2d(
                x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
                softwta_activs.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
                padding=0,
                stride=self.dilation,
                dilation=self.stride,
                groups=1
            ).transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)

            # sum over batch, output pixels: each kernel element will influence all batches and output pixels.
            yu = torch.sum(torch.mul(softwta_activs, weighted_input), dim=(0, 2, 3))
            delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight
            delta_weight.div_(torch.abs(delta_weight).amax() + 1e-30)  # Scale [min/max , 1]
            self.weight.grad = delta_weight  # store in grad to be used with common optimizers

        return weighted_input


class DeepSoftHebb(nn.Module):
    # DeepSoftHebb是包含SoftHebbConv2d层的深度卷积神经网络
    # DeepSoftHebb定义了包含SoftHebbConv2d层的深度卷积神经网络。
    # 网络由多个块组成，每个块包含SoftHebbConv2d层、激活函数、池化层等。最后有一个线性分类器。
    def __init__(self):
        super(DeepSoftHebb, self).__init__()
        # block 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = SoftHebbConv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2, t_invert=1,)
        self.activ1 = Triangle(power=0.7)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # block 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = SoftHebbConv2d(in_channels=96, out_channels=384, kernel_size=3, padding=1, t_invert=0.65,)
        self.activ2 = Triangle(power=1.4)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # block 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = SoftHebbConv2d(in_channels=384, out_channels=1536, kernel_size=3, padding=1, t_invert=0.25,)
        self.activ3 = Triangle(power=1.)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # block 4
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(24576, 10)
        self.classifier.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

        self.representation = None

    def forward(self, x):
        # forward函数定义了网络的前向传播过程，通过多个块进行卷积、激活和池化操作，最后通过线性分类器进行分类。
        # block 1
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))  # x -> bn1 -> conv1 -> activ1 -> pool1
        # block 2
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))  # out -> bn2 -> conv2 -> activ2 -> pool2
        # block 3
        out = self.pool3(self.activ3(self.conv3(self.bn3(out))))  # out -> bn3 -> conv3 -> activ3 -> pool3
        self.representation = self.flatten(out)
        # block 4
        return self.classifier(self.dropout(self.flatten(out)))  # out -> flatten -> classifier


class Triangle(nn.Module):
    # Triangle是一个激活函数模块，使用ReLU和指数幂进行激活。
    # 它通过将输入减去均值进行中心化，并应用ReLU激活和指数幂来实现。
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power


class WeightNormDependentLR(optim.lr_scheduler._LRScheduler):
    # WeightNormDependentLR是一个自定义的学习率调度器，根据神经元的范数与理论收敛范数之间的差异进行学习率缩放
    """
    Custom Learning Rate Scheduler for unsupervised training of SoftHebb Convolutional blocks.
    Difference between current neuron norm and theoretical converged norm (=1) scales the initial lr.
    """

    def __init__(self, optimizer, power_lr, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.initial_lr_groups = [group['lr'] for group in self.optimizer.param_groups]  # store initial lrs
        self.power_lr = power_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        new_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                # difference between current neuron norm and theoretical converged norm (=1) scales the initial lr
                # initial_lr * |neuron_norm - 1| ** 0.5
                norm_diff = torch.abs(torch.linalg.norm(param.view(param.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                new_lr.append(self.initial_lr_groups[i] * (norm_diff ** self.power_lr)[:, None, None, None])
        return new_lr


class TensorLRSGD(optim.SGD):
    # TensorLRSGD是一个自定义的带有张量学习率的SGD优化器。
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step, using a non-scalar (tensor) learning rate.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(-group['lr'] * d_p)
        return loss


class CustomStepLR(StepLR):
    """
    CustomStepLR是一个自定义的学习率调度器，根据训练的epoch数量调整学习率。
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


class FastCIFAR10(torchvision.datasets.CIFAR10):
    """
    FastCIFAR10是一个加速CIFAR10数据集加载的自定义数据集类，去除了PIL接口，并在GPU上进行预加载。

    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        super().__init__(*args, **kwargs)

        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)
        self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.data[index]
        target = self.targets[index]

        return img, target


# 这是CIFAR10数据集的主要训练循环。在循环中，首先进行SoftHebb的无监督训练，然后进行有监督训练，最后进行模型的评估。
# Main training loop CIFAR10
if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = DeepSoftHebb()
    model.to(device)

    # 非监督的优化器和学习率调度器
    unsup_optimizer = TensorLRSGD([
        {"params": model.conv1.parameters(), "lr": -0.08, },  # SGD does descent, so set lr to negative
        {"params": model.conv2.parameters(), "lr": -0.005, },
        {"params": model.conv3.parameters(), "lr": -0.01, },
    ], lr=0)
    unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)

    # 监督的优化器和学习率调度器
    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    sup_lr_scheduler = CustomStepLR(sup_optimizer, nb_epochs=50)
    criterion = nn.CrossEntropyLoss()

    trainset = FastCIFAR10('./data', train=True, download=True)  # Dataset FastCIFAR10 Number of datapoints: 50000 Root location: ./data Split: Train
    unsup_trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, )
    sup_trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, )

    testset = FastCIFAR10('./data', train=False)  # Dataset FastCIFAR10 Number of datapoints: 10000 Root location: ./data Split: Test
    # Use a fixed seed for the random number generator
    torch.manual_seed(42)
    # Ensure the same test data is used each time and the order is the same
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
    import pdb; pdb.set_trace()
    # Unsupervised training with SoftHebb
    running_loss = 0.0
    for i, data in enumerate(unsup_trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)

        # zero the parameter gradients
        unsup_optimizer.zero_grad()

        # forward + update computation
        with torch.no_grad():
            outputs = model(inputs)

        # optimize
        unsup_optimizer.step()
        unsup_lr_scheduler.step()

    # Supervised training of classifier
    # set requires grad false and eval mode for all modules but classifier
    unsup_optimizer.zero_grad()
    model.conv1.requires_grad = False
    model.conv2.requires_grad = False
    model.conv3.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    model.conv3.eval()
    model.bn1.eval()
    model.bn2.eval()
    model.bn3.eval()
    for epoch in range(50):
        model.classifier.train()
        model.dropout.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(sup_trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            sup_optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            sup_optimizer.step()

            # compute training statistics
            running_loss += loss.item()
            if epoch % 10 == 0 or epoch == 49:
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        sup_lr_scheduler.step()
        # Evaluation on test set
        if epoch % 10 == 0 or epoch == 49:
            print(f'Accuracy of the network on the train images: {100 * correct // total} %')
            print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

            # on the test set
            model.eval()
            running_loss = 0.
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            import pdb; pdb.set_trace()
            from tqdm import tqdm
            # with tqdm(total=len(testloader), desc=f'Testing Epoch {epoch}') as pbar:
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
            print(f'test loss: {running_loss / total:.3f}')


# modify the code to make sure that each time a test data is used, the same test data is used and the test data order is the same.
# record the activation of self.representation of the model when the test data is used.
# modify the code to save the activation of self.representation of the model when the test data is used.

