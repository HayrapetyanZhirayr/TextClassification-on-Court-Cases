import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

class LinearModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        if input_dim == output_dim:
            self.weight = nn.Parameter(
                        .1*torch.randn(input_dim, output_dim) +
                        torch.eye(input_dim, output_dim)
                    )
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
            self.bias = nn.Parameter(torch.zeros(output_dim))

            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, input):
        logits = input@self.weight + self.bias
        return logits


class CustomLinearLayer(nn.Module):
    def __init__(self, S):
        output_dim, input_dim = S.shape
        # input_dim :: number of words (features)
        # output_dim :: number of classes
        super().__init__()

        self.THETA = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.S = S

        # initialize weights correctly
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.THETA)


    def forward(self, input):
        batch_dim, input_dim = input.shape
        _input = input.view(batch_dim, 1, input_dim)*self.S
        # _input is [*, output_dim, input_dim]
        logits = (_input*self.THETA).sum(axis=-1) + self.bias
        return logits


class CustomModel(nn.Module):
    def __init__(self, S):
        output_dim, input_dim = S.shape
        # input_dim :: number of words (features)
        # output_dim :: number of classes
        super().__init__()

        # self.THETA = nn.Parameter(torch.randn(output_dim, input_dim))
        # self.bias = nn.Parameter(torch.zeros(output_dim))
        self.S = S

        self.custom_linear = CustomLinearLayer(self.S)
        self.custom_linear.reset_parameters()

        self.weight = nn.Parameter(
            .1*torch.randn(output_dim, output_dim) +
            torch.eye(output_dim, output_dim)
                )
        self.bias2 = nn.Parameter(torch.zeros(output_dim))

        # initialize weights correctly
        self.reset_parameters()

    def reset_parameters(self):
        pass


    def forward(self, input):
        batch_dim, input_dim = input.shape
        logits = self.custom_linear(input)
        # _input = input.view(batch_dim, 1, input_dim)*self.S
        # _input is [*, output_dim, input_dim]
        # logits = (_input*self.THETA).sum(axis=-1) + self.bias
        f = nn.functional.relu(logits)
        f2 = f@self.weight + self.bias2
        return f2


class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dim1=2**10):
        # input_dim :: number of words
        # output_dim :: number of classes
        super().__init__()

        # layer_dim1 = 512*2

        self.layer_dim1 = layer_dim1

        self.weight1 = nn.Parameter(torch.randn(input_dim, layer_dim1))
        self.bias1 = nn.Parameter(torch.zeros(layer_dim1))


        self.weight2 = nn.Parameter(torch.randn(layer_dim1, output_dim))
        self.bias2 = nn.Parameter(torch.zeros(output_dim))

        # initialize weight correctly
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight1)
        nn.init.kaiming_normal_(self.weight2)

    def forward(self, input):

        f1 = nn.functional.relu(input @ self.weight1 + self.bias1)
        f2 = f1 @ self.weight2 + self.bias2

        return f2


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dim1=2**8, layer_dim2=2**6):
        # input_dim :: number of words
        # output_dim :: number of classes
        super().__init__()

        self.layer_dim1 = layer_dim1

        self.weight1 = nn.Parameter(torch.randn(input_dim, layer_dim1))
        self.bias1 = nn.Parameter(torch.zeros(layer_dim1))

        self.layer_dim2 = layer_dim2

        self.weight2 = nn.Parameter(torch.randn(layer_dim1, layer_dim2))
        self.bias2 = nn.Parameter(torch.zeros(layer_dim2))



        self.weight3 = nn.Parameter(torch.randn(layer_dim2, output_dim))
        self.bias3 = nn.Parameter(torch.zeros(output_dim))

        # initialize weight correctly
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight1)
        nn.init.kaiming_normal_(self.weight2)

    def forward(self, input):

        f1 = nn.functional.relu(input @ self.weight1 + self.bias1)
        f2 = nn.functional.relu(f1 @ self.weight2 + self.bias2)
        f3 = f2 @ self.weight3 + self.bias3

        return f3


class SepConv1D(nn.Module):
    """
    pyTorch implementation of SeparableConv1D from keras
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SepConv1D, self).__init__()
        self.in_channels = in_channels
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
            padding="same", groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
            padding="same")

    def forward(self, x):
        _, in_channels, steps = x.shape
        if in_channels != self.in_channels and steps == self.in_channels:
            # batch, steps, in_channels -> batch, channels, steps
            x = x.permute(0, 2, 1)
        out = self.depthwise(x)
        out = self.pointwise(out)
        # output is batch, out_channels, steps
        return out



class SepConvBlock(nn.Module):
    """
    Block of Separable Convolutions used for google text classification model
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(SepConvBlock, self).__init__()
        self.sepconv1 = SepConv1D(in_channels, out_channels, kernel_size)
        self.relu1 = nn.ReLU()
        self.sepconv2 = SepConv1D(out_channels, out_channels, kernel_size)
        self.relu2 = nn.ReLU()
        self.mpool1D = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        out = self.sepconv1(x)
        out = self.relu1(out)
        out = self.sepconv2(out)
        out = self.relu2(out)
        out = self.mpool1D(out)
        return out

class SepConvModel(nn.Module):
    """
    Google Model of Separable Convolutions for text classification
    """
    def __init__(self, out_channels, kernel_size, pool_size,
        n_blocks, dropout_rate, embedding_dim, output_dim, num_embeddings):
        """
        # Arguments
            num_embeddings: int, size of vocabulary to embed
            output_dim: int, number of classes
        """
        super(SepConvModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        in_channels = embedding_dim
        self.n_blocks = n_blocks
        self.blocks = []
        for i in range(n_blocks - 1):
            setattr(self, "dropout" + str(i+1), nn.Dropout(p=dropout_rate))
            self.blocks.append(getattr(self, "dropout" + str(i+1)))


            setattr(self, "sepBlock" + str(i+1), SepConvBlock(in_channels, out_channels,
                    kernel_size, pool_size))
            self.blocks.append(getattr(self, "sepBlock" + str(i+1)))
            in_channels = out_channels

        self.lastblocksep1 = SepConv1D(out_channels, out_channels*2, kernel_size)
        self.relu1 = nn.ReLU()
        self.lastblocksep2 = SepConv1D(out_channels*2, out_channels*2,
                kernel_size)
        self.relu2 = nn.ReLU()

        self.gpool = lambda x: nn.AvgPool1d(kernel_size=x)

        self.dropout_last = nn.Dropout(p=dropout_rate)

        self.dense = nn.Linear(out_channels*2, output_dim)

    def forward(self, x):
        # print(f"INPUT SHAPE :: {x.shape}")
        out = self.embedding(x)
        # print(f"EMBEDDING SHAPE :: {out.shape}")

        for layer in self.blocks:
            out = layer(out)

        # print(f"AFTER SEP BLOCKS SHAPE :: {out.shape}")

        out = self.lastblocksep1(out)
        out = self.relu1(out)
        # print(f"AFTER lastblocksep1 SHAPE :: {out.shape}")


        out = self.lastblocksep2(out)
        out = self.relu2(out)
        # print(f"AFTER lastblocksep2 SHAPE :: {out.shape}")

        kernel_size = out.shape[-1]
        out = self.gpool(kernel_size)(out).squeeze()
        out = self.dropout_last(out)
        # print(f"AFTER GPOOL SHAPE :: {out.shape}")


        out = self.dense(out)
        # print(f"OUTPUT SHAPE :: {out.shape}")

        return out









# auxiliary stuff
class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
