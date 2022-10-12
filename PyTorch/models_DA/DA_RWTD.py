import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .miniViT import mViT


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha
        # Forward pass is a no-op
        return x.view_as(x) # view_as return the same scale tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha # .neg() Returns a new tensor with the negative of the elements of input
        # Must return same number as inputs to forward()
        return output, None


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(64 * 128 * 256, 128)
        self.fc2 = nn.Linear(128, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, grl_lambda):

        # domain_features = x[-1].view(-1, 2048 * 8 * 16)  # features[-1]: torch.Size([bs, 2048, 8, 16])
        domain_features = x.view(-1, 64 * 128 * 256)  # features[-1]: torch.Size([bs, 64, 128, 256])
        reverse_features = GradientReversalFn.apply(domain_features, grl_lambda)

        out = self.fc1(reverse_features)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out



class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2) # expected input[8, 127, 128, 256]
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class DA_RWTD(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(DA_RWTD, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        # embbeded features from encoder
        self.encoder = Encoder(backend)
        self.bins_layer = mViT(64, n_query_channels=64, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.decoder = DecoderBN(num_classes=64)
        self.discriminator = Discriminator()
        self.conv_out = nn.Sequential(nn.Conv2d(64, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, grl_lambda, **kwargs):
        # output from the first module
        unet_out = self.decoder(self.encoder(x), **kwargs) # [bs, num_classes, h/2, w/2] --> torch.Size([8, num_classes, 208, 272]) --> torch.Size([8, 64, 128, 256])

        ####################################################################################
        #                               ONLY reverse layer
        ####################################################################################
        domain_features = unet_out.view(-1, 64 * 128 * 256)  # features[-1]: torch.Size([bs, 64, 128, 256])
        reverse_features = GradientReversalFn.apply(domain_features, grl_lambda)
        pred_domain_label = reverse_features

        # bin_widths_normed --> torch.Size([8, 80])
        # range_attention_maps --> same size as unet_out --> torch.Size([8, 64, 128, 256])
        bin_widths_normed, range_attention_maps = self.bins_layer(unet_out)
        out = self.conv_out(range_attention_maps) # torch.Size([8, 80, 128, 256])

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1) # torch.Size([8, 80, 1, 1])

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred, pred_domain_label

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m

