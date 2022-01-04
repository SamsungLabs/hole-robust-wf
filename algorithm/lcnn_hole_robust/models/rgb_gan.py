import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lcnn.config import C, M

        
class RGBGANLoss(nn.Module):
    def __init__(self):
        super(RGBGANLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pretrained_models = list()
        for pretrained_model_features in [[0] + M.pretrained_model_features_per, [0] + M.pretrained_model_features_sty]:
            pretrained_model = list()
            pretrained_model_load = torchvision.models.vgg19(pretrained=True).features
            for features_idx in range(1, len(pretrained_model_features)):
                pretrained_model.append(nn.Sequential())
                for idx in range(pretrained_model_features[features_idx - 1], pretrained_model_features[features_idx]):
                    pretrained_model[-1].add_module(str(idx), pretrained_model_load[idx].to(self.device))                    
            self.pretrained_models.append(pretrained_model)

    @staticmethod
    def _gram_matrix(feature):
        F_ik = feature.reshape(feature.shape[0], feature.shape[1], -1)
        F_jk = feature.reshape(feature.shape[0], feature.shape[1], -1).permute(0, 2, 1)
        G_ij = torch.div(torch.bmm(F_ik, F_jk), torch.prod(torch.tensor(list(feature.shape[1:]))))
        return G_ij

    @staticmethod
    def _predict(pretrained_model, layer, input):
        out = dict()
        for idx, layer in enumerate(layer):
            input = pretrained_model[idx](input)
            out[layer] = input

        return out

    def loss_adv_dis(self, pred, gt):
        bce = nn.BCELoss()
        loss = bce(pred, torch.zeros_like(pred).to(self.device)) + bce(gt, torch.ones_like(gt).to(self.device))

        return loss

    def loss_adv_gen(self, pred):   # non-saturating version
        loss = nn.BCELoss()(pred, torch.ones_like(pred).to(self.device))

        return loss

    def loss_per(self, pred, gt, reduction="mean"):
        feat_pred = self._predict(self.pretrained_models[0], M.pretrained_model_features_per, pred)
        feat_gt = self._predict(self.pretrained_models[0], M.pretrained_model_features_per, gt)

        loss = 0.0
        for layer in M.pretrained_model_features_per:
            loss += nn.L1Loss(reduction=reduction)(feat_pred[layer], feat_gt[layer])

        return loss

    def loss_sty(self, pred, gt, reduction="mean"):
        feat_pred = self._predict(self.pretrained_models[1], M.pretrained_model_features_sty, pred)
        feat_gt = self._predict(self.pretrained_models[1], M.pretrained_model_features_sty, gt)

        loss = 0.0
        for layer in M.pretrained_model_features_sty:
            loss += nn.L1Loss(reduction=reduction)(self._gram_matrix(feat_pred[layer]), self._gram_matrix(feat_gt[layer]))

        return loss       

    def loss_rec(self, pred, gt, mask=None, reduction="none"):
        loss = F.l1_loss(pred, gt, reduction=reduction)

        if mask is not None:
            w = torch.mean(mask, dim=(1,2,3), keepdim=True)
            w[w == 0] = 1
            loss = loss * (mask / w)

        loss = torch.mean(loss, dim=(1,2,3), keepdim=False)

        return loss


def spectral_norm(module, use_spectral_norm=True, n_power_iterations=1, eps=1e-12, dim=None):
    return nn.utils.spectral_norm(module, n_power_iterations=n_power_iterations, eps=eps, dim=dim) if use_spectral_norm else module


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    BSD License
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class NLayerDiscriminator(nn.Module):
    """
    BSD License
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            (our customization) use_spectral_norm -- if true, use spectral norm after each convolution
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = not use_spectral_norm
        self.n_layers = n_layers

        kw = 4
        padw = 1
        sequence = [[spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_spectral_norm),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[spectral_norm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_spectral_norm)]]  # output 1 channel prediction map

        for idx in range(len(sequence)):
            setattr(self, 'layer'+str(idx), nn.Sequential(*sequence[idx]))

        init_weights(self)

    def forward(self, x):
        feat = [x]
        for idx in range(self.n_layers + 2):
            layer = getattr(self, 'layer' + str(idx))
            feat.append(layer(feat[-1]))        
        out = torch.sigmoid(feat[-1])
        return out, feat[1:]


class RGBGANDiscriminatorOptimizer(nn.Module):
    def __init__(self, model, lr):
        super(RGBGANDiscriminatorOptimizer).__init__()
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.0, 0.9))


class RGBGANDiscriminatorLearner(nn.Module):
    def __init__(self):
        super(RGBGANDiscriminatorLearner, self).__init__()

        self.rgb_gan_losses = RGBGANLoss()
        self.use_spectral_norm = M.use_spectral_norm and C.io.is_rgb_gan
        self.model_d = NLayerDiscriminator(3, use_spectral_norm=self.use_spectral_norm)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mean = torch.tensor([M.image.mean]).unsqueeze(2).unsqueeze(3).to(self.device)
        self.std = torch.tensor([M.image.stddev]).unsqueeze(2).unsqueeze(3).to(self.device)        

    def forward(self, input_dict, result, is_d=False, is_g=False):
        # <image, mask>
        image = (input_dict["image_gt"] * self.std + self.mean) / 255.0     # torch.Size([batch_size, 3, 512, 512])
        mask = input_dict["image"][:,3:4,:,:]                               # torch.Size([batch_size, 1, 512, 512])
        image = F.interpolate(image, size=(128,128), mode='area')           # torch.Size([batch_size, 3, 128, 128])
        mask = F.interpolate(mask, size=(128,128), mode='nearest')          # torch.Size([batch_size, 1, 128, 128])

        result_rgb_gan_losses = dict()

        if is_d:
            input_real_d = image
            input_fake_d = result["image_pred"].detach()

            output_real_d, _ = self.model_d(input_real_d)
            output_fake_d, _ = self.model_d(input_fake_d)

            result_rgb_gan_losses["losses_dis"] = M.loss_weight["dis"] * self.rgb_gan_losses.loss_adv_dis(output_fake_d, output_real_d)

        if is_g:
            input_real_g = image
            input_fake_g = result["image_pred"]

            output_fake_g, _ = self.model_d(input_fake_g)

            result_rgb_gan_losses["losses_gen"] = M.loss_weight["adv"] * self.rgb_gan_losses.loss_adv_gen(output_fake_g)
            result_rgb_gan_losses["losses_per"] = M.loss_weight["per"] * self.rgb_gan_losses.loss_per(input_fake_g, input_real_g)
            result_rgb_gan_losses["losses_sty"] = M.loss_weight["sty"] * self.rgb_gan_losses.loss_sty(input_fake_g*mask, input_real_g*mask)

        return result_rgb_gan_losses


class RGBGANHourglassDecoderLearner(nn.Module):
    def __init__(self):
        super(RGBGANHourglassDecoderLearner, self).__init__()

        self.rgb_gan_losses = RGBGANLoss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mean = torch.tensor([M.image.mean]).unsqueeze(2).unsqueeze(3).to(self.device)
        self.std = torch.tensor([M.image.stddev]).unsqueeze(2).unsqueeze(3).to(self.device)

    def forward(self, input_dict, image_pred):
        image_gt = (input_dict["image_gt"] * self.std + self.mean) / 255.0  # torch.Size([batch_size, 3, 512, 512])
        mask = input_dict["image"][:,3:4,:,:]                               # torch.Size([batch_size, 1, 512, 512])

        image_gt = F.interpolate(image_gt, size=(128,128), mode="area")     # torch.Size([batch_size, 3, 128, 128])
        mask = F.interpolate(mask, size=(128,128), mode="nearest")          # torch.Size([batch_size, 1, 128, 128])

        result = dict()
        result["losses_rec"] = M.loss_weight["rec"] * self.rgb_gan_losses.loss_rec(image_pred, image_gt, mask=1-mask)        
        result["image_pred"] = image_pred

        return result
