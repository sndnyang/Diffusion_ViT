import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        self.conv_layers = nn.Conv2d(n_input_channels, n_output_channels, kernel_size=(kernel_size, kernel_size),
                                     stride=(stride, stride), padding=(padding, padding), bias=conv_bias)
        self.act = nn.Identity() if activation is None else activation()
        self.pool = nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride,
                                 padding=pooling_padding) if max_pool else nn.Identity()

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.act(x)
        x = self.pool(x)
        return self.flattener(x).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    images = torch.randn((8, 3, 96, 96))
    kernel_size = 8
    tokenizer = Tokenizer(n_input_channels=3,
                          n_output_channels=392,
                          kernel_size=kernel_size,
                          stride=kernel_size,
                          padding=0,
                          max_pool=False,
                          activation=None,
                          conv_bias=True
                          )
    import time
    start = time.time()
    output = tokenizer(images)
    end = time.time()
    print(start - end)
    print(output.shape)
    patch_height, patch_width = 8, 8

    print(12 * 12 * 3)
    to_patch_embedding = nn.Sequential(
        Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=patch_height,
            p2=patch_width),
        nn.Linear(192, 392)
    )
    start = time.time()
    out2 = to_patch_embedding(images)
    print(out2.shape)
    end = time.time()
    print(start - end)
