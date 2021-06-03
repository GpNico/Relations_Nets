# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn import init

import torchvision

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
    def __init__(self, conf, height, width, dim_points):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)

        self.dim_z = 64

        self.encoder = EncoderNet(height, width,  dim_out = 2*self.dim_z)
        
        self.MLP = nn.Sequential(nn.Linear(self.dim_z, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, dim_points),
                                 nn.Sigmoid())
        
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
        
        latent_vectors = []

        for i, mask in enumerate(masks):
            z = self.__encoder_step(x, mask)
            latent_vectors.append(z)
            outputs.append(self.MLP(z))
            
        outputs = torch.stack(outputs, dim = 1)
        latent_vectors = torch.stack(latent_vectors, dim = 1)
        masks = torch.cat(masks, 1)
        
        dict = {'outputs_slot': outputs, 'masks': masks, 'latent_vectors': latent_vectors}
  
        return dict
        
    def get_loss(self, x, target, loss_function):
        dict = self.forward(x)
        outputs = dict['outputs_slot']
        carac_labels = target['carac_labels']
        loss, _ = loss_function(outputs, carac_labels)
        return dict, loss


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :self.dim_z]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, self.dim_z:]) * 3
        sigmas += 1e-5*torch.ones_like(sigmas)
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

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        init.xavier_uniform_(self.slots_mu)

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim,  bias=True)
        self.to_k = nn.Linear(dim, dim,  bias=True)
        self.to_v = nn.Linear(dim, dim,  bias=True)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim, eps=0.001)
        self.norm_slots  = nn.LayerNorm(dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001)

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            masks = attn

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, masks


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
    def __init__(self, conf, height, width, dim_points):
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
                                            iters = 7)

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

        slots, attn = self.slot_attention(x)
        # x.shape [batch_size, num_slots, slot_size]

        predictions = self.mlp_classifier(slots)
        
        dict = {'outputs_slot': predictions, 'masks': attn, 'latent_vectors': slots}

        return dict
        
    def get_loss(self, x, target, loss_function):
        dict = self.forward(x)
        outputs = dict['outputs_slot']
        carac_labels = target['carac_labels']
        loss, _ = loss_function(outputs, carac_labels)
        return dict, loss
        
        
##############################################################################################################
##                                                                                                          ##
##                                         RELATIONS PREDICTOR                                              ##
##                                                                                                          ##
##############################################################################################################       


class RelationsPredictor(nn.Module):
    def __init__(self, conf, height, width, dim_points, dim_rela, object_classifier = 'slot_att'):
        super().__init__()
        
        self.conf = conf
        
        self.alpha = 0.
        self.step = 0
        self.N_alpha = 20000
        self.epsilon_alpha = 10**(-5)
        
        if object_classifier == 'monet':
            self.obj_class = MonetClassifier(conf, height, width, dim_points)
        elif object_classifier == 'slot_att':
            self.obj_class = SlotAttentionClassifier(conf, height, width, dim_points)
            
        
        self.mlp_rela = nn.Sequential(nn.Linear(2*64, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, dim_rela),
                                            nn.Sigmoid())    
            
        
            
    def forward(self, images):
        
        dict = self.obj_class(images)
        
        latent_vectors = dict['latent_vectors']
        
        _, num_slots, _ = latent_vectors.shape
        
        outputs_rela = []
        index_to_pair = []
        
        for k in range(num_slots):
            for l in range(k):
                z_k, z_l = latent_vectors[:, k, :], latent_vectors[:, l, :]
                z_kl = torch.cat((z_k, z_l), dim = -1)
                z_lk = torch.cat((z_l, z_k), dim = -1)
                
                rela_kl = self.mlp_rela(z_kl)
                rela_lk = self.mlp_rela(z_lk)
                
                outputs_rela.append(rela_kl)
                outputs_rela.append(rela_lk)
                
                index_to_pair.append([k, l])
                index_to_pair.append([l, k])
        
        outputs_rela = torch.stack(outputs_rela, dim = 1)
        
        dict['outputs_rela'] = outputs_rela
        dict['index_to_pair'] = index_to_pair
        
        return dict
        
        
    def get_loss(self, x, target, loss_function):
        
        dict = self.forward(x)
        
        outputs_slot = dict['outputs_slot'] #[batch_size, n_slots, dim_points]
        outputs_rela = dict['outputs_rela'] #[batch_size, n_slots*(n_slots - 1), dim_rela]
        index_to_pair = dict['index_to_pair'] #[n_slots*(n_slots - 1)]
        
        carac_labels = target['carac_labels'] #[batch_size, n_slots, dim_points]
        rela_labels = target['rela_labels'] #[batch_size, n_slots, n_slots, dim_rela]
        
        loss_obj_class, sigmas = loss_function(outputs_slot, carac_labels)
        
        #Computing sigma_inv
        sigmas_inv = torch.zeros_like(sigmas).cuda()
        batch_size, num_slots = sigmas.shape
        for k in range(batch_size):
            for l in range(num_slots):
                sigmas_inv[k, sigmas[k, l]] = l
        
        #fundamental relation : outputs_rela[n, i] ~ rela_labels[n, sigma_inv[index_to_pair[i][0]], sigma_inv[index_to_pair[i][1]]]
        
        loss_function_2 = torch.nn.SmoothL1Loss(reduction = 'none')
        
        loss_rela_class = 0.
        
        for i in range(num_slots*(num_slots-1)):
            loss_rela_class += torch.mean(loss_function_2(outputs_rela[:, i], 
                                                         rela_labels[np.arange(batch_size), sigmas_inv[:,index_to_pair[i][0]], sigmas_inv[:,index_to_pair[i][1]]]),
                                         dim = -1)
        loss_rela_class = torch.mean(loss_rela_class)
        
        #add the losses
        
        loss = loss_obj_class + self.alpha * loss_rela_class

        #Update Alpha
        
        self.step += 1
        if self.step > self.N_alpha:
           self.alpha += self.epsilon_alpha
       
        return dict, loss

































