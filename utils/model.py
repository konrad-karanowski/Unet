import torch
from torch import nn


class UNet(nn.Module):
    """
    UNet image segmentation model
    Based on: https://arxiv.org/pdf/1505.04597.pdf
    Small adjustment (?) is that input images and output masks are the same size (256 x 256).

    Input:
    - torch.Tensor with shape (batch size, number of channels, 256, 256)
    Every input [i, j] (where i is the number of image in the batch and j is the certain channel)
    is an image with numbers from range[0, 255]

    Output:
    - torch.Tensor with shape(batch size, number of classes, 256, 256)
    Every output[i, j] (where i is the number of image in the batch and j is the certain class) is a mask
    with numbers from range[0., 1.]. Values can be  interpreted as degrees of belonging to certain class.
    Suggested threshold for which we want to classify pixel as belonging to class is 0.5 +.

    We can interpret mask as type I fuzzy set

    Layers:
    Algorithm is build with 3 major parts
    a) encoding path:
    b) decoding path:
    c) final output:

    Total trainable parameters: 31 043 586
    Total non-trainable parameters: 0
    """

    def __init__(self, num_channels: int, num_classes: int):
        """
        Initialize all layers used in algorithm
        :param num_channels: number of channels in image (1 for grays and 3 for colored ones)
        :param num_classes: number of classes
        """
        super().__init__()
        self.__input_size = (num_channels, 256, 256)
        # pooling
        self.__max_pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder layers
        self.__double_conv1down = self.__double_conv(num_channels, 64)
        self.__double_conv2down = self.__double_conv(64, 128)
        self.__double_conv3down = self.__double_conv(128, 256)
        self.__double_conv4down = self.__double_conv(256, 512)
        self.__double_conv5down = self.__double_conv(512, 1024)

        # upsampling [input_channels, output_channels, kernel_size, stride]
        self.__upsample_conv5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.__upsample_conv4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.__upsample_conv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.__upsample_conv2 = nn.ConvTranspose2d(128, 64, 2, 2)

        # decoder layers
        self.__double_conv4up = self.__double_conv(1024, 512)
        self.__double_conv3up = self.__double_conv(512, 256)
        self.__double_conv2up = self.__double_conv(256, 128)
        self.__double_conv1up = self.__double_conv(128, 64)

        # output layer
        self.__output = nn.Conv2d(64, num_classes, kernel_size=1)
        self.__activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward-pass
        :param x: input tensor of size (batch size, number of channels, 256, 256)
        :return: output mask tensor of size (batch_size, number of classes, 256, 256)
        """
        # encoding
        x1_conv_down = self.__double_conv1down(x)
        x1_out_down = self.__max_pool2x2(x1_conv_down)
        x2_conv_down = self.__double_conv2down(x1_out_down)
        x2_out_down = self.__max_pool2x2(x2_conv_down)
        x3_conv_down = self.__double_conv3down(x2_out_down)
        x3_out_down = self.__max_pool2x2(x3_conv_down)
        x4_conv_down = self.__double_conv4down(x3_out_down)
        x4_out_down = self.__max_pool2x2(x4_conv_down)
        x5_conv_down = self.__double_conv5down(x4_out_down)

        # decoding path
        x4_upsample = self.__upsample_conv5(x5_conv_down)
        x4_cat_up = self.__concat(x4_conv_down, x4_upsample)
        x4_conv_up = self.__double_conv4up(x4_cat_up)

        x3_upsample = self.__upsample_conv4(x4_conv_up)
        x3_cat_up = self.__concat(x3_conv_down, x3_upsample)
        x3_conv_up = self.__double_conv3up(x3_cat_up)

        x2_upsample = self.__upsample_conv3(x3_conv_up)
        x2_cat_up = self.__concat(x2_conv_down, x2_upsample)
        x2_conv_up = self.__double_conv2up(x2_cat_up)

        x1_upsample = self.__upsample_conv2(x2_conv_up)
        x1_cat_up = self.__concat(x1_conv_down, x1_upsample)
        x1_conv_up = self.__double_conv1up(x1_cat_up)

        # output
        output = self.__output(x1_conv_up)
        return self.__activation(output)

    def __concat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Concatenate 2 tensors by the dimension of 1 (number of channels)
        :param x1: tensor of size [a, X, b, c]
        :param x2: tensor of size [a, Y, b, c]
        :return: tensor of size [a, X + Y, b, c]
        """
        return torch.cat((x1, x2), dim=1)

    def __double_conv(self, input_channels: int, output_channels: int,
                      kernel_size: int = 3) -> nn.modules.container.Sequential:
        """
        Creates double convolution layer with ReLU as activation function and batch normalization
        :param input_channels: number of channels coming to the double convolution layer
        :param output_channels: number of channels coming out of the double convolution layer
        :param kernel_size: kernel stride for convolution network
        :return: double convolution layer
        """
        layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channels)
        )
        return layer

    @property
    def input_size(self) -> (int, int, int):
        """
        Returns input size (required for torchsummary)
        :return: three elements tuple of elements (number of channels, 256, 256)
        """
        return self.__input_size


def setup_model(input_channels: int, num_classes: int, model_path: str) -> UNet:
    """
    Prepares UNet model for training
    :param input_channels: input channels (1 for grays and 3 for coloured ones)
    :param num_classes: number of classes
    :param model_path: (optional) path to model, if user wants to use pretrained one
    :return: UNet model
    """
    model = UNet(input_channels, num_classes)
    if model_path:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    return model
