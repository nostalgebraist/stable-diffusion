import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


def hinge_abs_loss(t1, t2, cut):
    return F.relu(torch.abs(t1 - t2) - cut)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge",
                 gen_start=0, beta1=0.5, beta2=0.9,
                 use_d=True,
                 use_original_sum_calc=False,
                 scale_g_loss=False,
                 reduce_all=False,
                 hinge_cut=0.0,
                 r1_weight=0.0,
                 ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.use_d = use_d

        if self.use_d:
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                     n_layers=disc_num_layers,
                                                     use_actnorm=use_actnorm
                                                     ).apply(weights_init)
        self.generator_iter_start = gen_start
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.use_original_sum_calc = use_original_sum_calc
        self.scale_g_loss = scale_g_loss
        self.reduce_all = reduce_all

        self.hinge_cut = hinge_cut
        self.r1_weight = r1_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None, g_scale_factor=1.0):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        gnorm_nll = torch.norm(nll_grads)
        gnorm_g = torch.norm(g_grads)
        gnorm_g = g_scale_factor * torch.norm(g_grads).detach().float()
        d_weight = gnorm_nll / (gnorm_g + 1e-4)
        d_weight = torch.isfinite(d_weight).float() * d_weight
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight, gnorm_nll.detach(), gnorm_g.detach()

    def calculate_r1_penalty(self, images_real, logits_real):
        gradients = torch.autograd.grad(outputs=logits_real.float(),
                                        inputs=images_real,
                                        grad_outputs=torch.ones(logits_real.size(), device=images_real.device),
                                        create_graph=True)[0]

        gradients = gradients.reshape(images_real.shape[0], -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() / 2.

        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return grad_norm / 2.

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        if self.hinge_cut > 0.:
            pixel_rec_loss = hinge_abs_loss(inputs.contiguous(), reconstructions.contiguous(), self.hinge_cut)
        else:
            pixel_rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        else:
            p_loss = torch.zeros_like(pixel_rec_loss)

        rec_loss = pixel_rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = nll_loss.float()
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss

        kl_loss = posteriors.kl()

        scale_factor = 1.0
        if not self.use_original_sum_calc:
            # reproduce sum calc for 256x256 res but auto scale loss terms at other sizes
            scale_factor = (256 / nll_loss.shape[-1]) ** 2

        g_scale_factor = 1.0
        if self.scale_g_loss:
            g_scale_factor = scale_factor * (nll_loss.shape[1] * nll_loss.shape[2] * nll_loss.shape[3])

        if self.reduce_all:
            scale_factor = 1.0
            g_scale_factor = 1.0
            kl_scale_factor = 1. / (nll_loss.shape[1] * nll_loss.shape[2] * nll_loss.shape[3])

            nll_loss = scale_factor * torch.mean(nll_loss)
            weighted_nll_loss = scale_factor * torch.mean(weighted_nll_loss)
            kl_loss = kl_scale_factor * scale_factor * torch.mean(kl_loss)
        else:
            nll_loss = scale_factor * torch.sum(nll_loss) / nll_loss.shape[0]
            weighted_nll_loss = scale_factor * torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            kl_loss = scale_factor * torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update

            if not self.use_d:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0)
            else:
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(reconstructions.contiguous())
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                logits_fake = logits_fake.float()
                g_loss = -torch.mean(logits_fake)

                if self.disc_factor > 0.0:
                    try:
                        d_weight, gnorm_nll, gnorm_g = self.calculate_adaptive_weight(
                            nll_loss, g_loss, last_layer=last_layer, g_scale_factor=g_scale_factor,
                        )
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0)
                        gnorm_nll = torch.tensor(0.0)
                        gnorm_g = torch.tensor(0.0)
                else:
                    d_weight = torch.tensor(0.0)
                    gnorm_nll = torch.tensor(0.0)
                    gnorm_g = torch.tensor(0.0)

            g_loss = g_scale_factor * g_loss

            gen_factor = adopt_weight(1.0, global_step, threshold=self.generator_iter_start)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = gen_factor * (weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss)

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/d_gnorm_nll".format(split): gnorm_nll.detach(),
                   "{}/d_gnorm_g".format(split): gnorm_g.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/rec_loss_pixel".format(split): pixel_rec_loss.detach().mean(),
                   "{}/rec_loss_prcpt".format(split): p_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                images_real = inputs.contiguous()
                images_real.requires_grad_(self.r1_weight > 0)
                logits_real = self.discriminator(images_real)
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                images_real = torch.cat((inputs.contiguous(), cond), dim=1)
                images_real.requires_grad_(self.r1_weight > 0)
                logits_real = self.discriminator(images_real)
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            logits_real, logits_fake = logits_real.float(), logits_fake.float()

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss_base = disc_factor * self.disc_loss(logits_real, logits_fake)

            d_loss = d_loss_base
            penalty = 0.0

            if self.r1_weight > 0:
                penalty = self.r1_weight * self.calculate_r1_penalty(images_real, logits_real)
                d_loss = d_loss + penalty

            log = {"{}/disc_loss".format(split): d_loss_base.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            if self.r1_weight > 0.0:
               log["{}/r1_penalty".format(split)] = penalty.clone().detach().mean()
            return d_loss, log
