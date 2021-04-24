#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: bert_pipeline.py.py

@time: 2021/02/01 10:30

@desc: Virtual adversarial training

"""
import torch
import torch.nn.functional as F


def kl(inputs, targets, reduction="sum"):
    loss = F.kl_div(F.log_softmax(inputs, dim=-1),
                    F.softmax(targets, dim=-1),
                    reduction=reduction)
    return loss


def adv_project(grad, norm_type='inf', eps=1e-6):
    if norm_type == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
    elif norm_type == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
    return direction


def virtual_adversarial_training(model, hidden_status, token_type_ids, attention_mask, logits):
    embed = hidden_status
    noise = embed.data.new(embed.size()).normal_(0, 1) * 1e-5
    noise.requires_grad_()
    new_embed = embed.data.detach() + noise
    adv_output = model(inputs_embeds=new_embed,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask)
    logits = logits
    adv_logits = adv_output[0]
    adv_loss = kl(adv_logits, logits.detach(), reduction="batchmean")
    delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
    norm = delta_grad.norm()

    if torch.isnan(norm) or torch.isinf(norm):
        return None

    # line 6 inner sum
    noise = noise + delta_grad * 1e-3
    # line 6 projection
    noise = adv_project(noise, norm_type='l2', eps=1e-6)
    new_embed = embed.data.detach() + noise
    new_embed = new_embed.detach()
    adv_output = model(inputs_embeds=new_embed,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask)
    adv_logits = adv_output[0]
    adv_loss_f = kl(adv_logits, logits.detach())
    adv_loss_b = kl(logits, adv_logits.detach())
    # 在预训练时设置为10，下游任务设置为1
    adv_loss = (adv_loss_f + adv_loss_b) * 1

    return adv_loss
