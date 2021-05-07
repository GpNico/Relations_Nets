# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.distributions as dists

import torchvision

from slot_attention import SlotAttention

import numpy as np


##############################################################################################################
##                                                                                                          ##
##                                                 MONET                                                    ##
##                                                                                                          ##
##############################################################################################################


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)


class AttentionNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = UNet(num_blocks=conf['num_blocks'],
                         in_channels=4,
                         out_channels=2,
                         channel_base=conf['channel_base'])

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope

class EncoderNet(nn.Module):
    def __init__(self, width, height, dim_out = 32):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim_out)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self, height, width, dim_in = 16):
        super().__init__()
        self.height = height
        self.width = width

        self.ch_1 = dim_in + 2

        self.convs = nn.Sequential(
            nn.Conv2d(self.ch_1, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf

        self.dim_z = 16

        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width, dim_out = 2*self.dim_z)
        self.decoder = DecoderNet(height, width, dim_in = self.dim_z)
        self.beta = 0.5
        self.gamma = 0.25

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.conf['num_slots']-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            sigma = self.conf['bg_sigma'] if i == 0 else self.conf['fg_sigma']
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        masks = torch.cat(masks, 1)
        tr_masks = torch.transpose(masks, 1, 3)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        return {'loss': loss,
                'masks': masks,
                'reconstructions': full_reconstruction}


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :self.dim_z]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, self.dim_z:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred
        
        
class MonetClassifier(nn.Module):
    def __init__(self, conf, height, width, dim_points, dim_rela = 0):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)

        self.dim_z = 64

        self.encoder = EncoderNet(height, width,  dim_out = 2*self.dim_z)
        
        self.MLP = nn.Sequential(nn.Linear(self.dim_z, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, dim_points),
                                 nn.Sigmoid())
                                       
        #Flag for relationship prediction
        if dim_rela > 0:
            self.pred_rela = True
            self.dim_rela = dim_rela
        else:
            self.pred_rela = False
            
        if self.pred_rela:
            self.MLP_binary_rela = nn.Sequential(nn.Linear(2*self.dim_z, 64),
                                                 nn.ReLU(),
                                                 nn.Linear(64, dim_rela))
        
        self.beta = 0.5
        self.gamma = 0.25

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.conf['num_slots']-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        
        outputs = []

        for i, mask in enumerate(masks):
            z = self.__encoder_step(x, mask)
            outputs.append(self.MLP(z))
            
        outputs = torch.stack(outputs, dim = 1)
        masks = torch.cat(masks, 1)
  
        return outputs, masks
        
    def get_loss(self, x, target, loss_function):
        outputs, _ = self.forward(x)
        loss = loss_function(outputs, target)
        return outputs, loss


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :self.dim_z]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, self.dim_z:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z


def print_image_stats(images, name):
    print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
    print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
    print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())



##############################################################################################################
##                                                                                                          ##
##                                             SLOT ATTENTION                                               ##
##                                                                                                          ##
##############################################################################################################


def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)



class SoftPositionEmbed(nn.Module):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.
    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()
    self.dense = nn.Linear(4, hidden_size)
    self.grid = torch.from_numpy(build_grid(resolution)).cuda()

  def forward(self, inputs):
    return torch.transpose(inputs, 3, 1) + self.dense(self.grid)


class SlotAttentionClassifier(nn.Module):
    def __init__(self, conf, height, width, dim_points, dim_rela = 0):
        super().__init__()
        self.conf = conf

        self.encoder_cnn = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 5, stride=1, padding = 2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, kernel_size = 5, stride=2, padding = 2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, kernel_size = 5, stride=2, padding = 2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, kernel_size = 5, stride=1, padding = 2),
                                         nn.ReLU(inplace=True)
                                        )

        resolution = (height//4, width//4) #Due to CNN
        
        self.encoder_pos = SoftPositionEmbed(64, resolution)

        self.layer_norm = torch.nn.LayerNorm(64, eps=0.001)

        self.mlp1 = nn.Sequential(nn.Linear(64, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 64))

        self.slot_attention = SlotAttention(num_slots = conf['num_slots'],
                                            dim = 64,
                                            iters = 3)

        self.mlp_classifier = nn.Sequential(nn.Linear(64, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, dim_points),
                                            nn.Sigmoid())

    def forward(self, image):

        x = self.encoder_cnn(image)
        x = self.encoder_pos(x) 
        # x.shape [batch_size, 32, 32, 64]

        x = x.reshape(-1, x.shape[1]*x.shape[2], x.shape[-1])
        # x.shape [batch_size, 32*32, 64]

        x = self.mlp1(self.layer_norm(x))
        # x.shape [batch_size, 32*32, 64]

        slots = self.slot_attention(x)
        # x.shape [batch_size, num_slots, slot_size]

        predictions = self.mlp_classifier(slots)

        return predictions, None
        
    def get_loss(self, x, target, loss_function):
        outputs, _ = self.forward(x)
        loss = loss_function(outputs, target)
        return outputs, loss