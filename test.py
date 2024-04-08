import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import *
from utils.general import *


def validation(args, model, dataloader):
    device = args.device
    contrastive_loss = nn.TripletMarginLoss(margin=1.0)
    consistency_loss = nn.MSELoss()

    val_loss = AverageMeter()
    val_mae_loss = AverageMeter()
    val_cont_loss = AverageMeter()
    val_cons_loss = AverageMeter()
    val_kl_loss = AverageMeter()
    val_num_slots = AverageMeter()
    val_sparsity = AverageMeter()

    model.eval()
    for x, _ in dataloader:
        x = x.unsqueeze(-1).to(device)
        
        with torch.no_grad():
            output, m_t, query, pos, neg, att_score, adjs = model(x)
            activated_memory_slots = len(torch.unique(torch.argmax(att_score, dim=-1)))
            d1 = torch.sum(adjs[0]) / args.num_nodes**2
            d2 = torch.sum(adjs[1]) / args.num_nodes**2
            sparsity = (1-d1 + 1-d2) / 2

            loss1 = masked_mae_loss(output, x)
            loss2 = contrastive_loss(query, pos.detach(), neg.detach())
            loss3 = consistency_loss(query, pos.detach())
            aggr_att_score = F.log_softmax(att_score.sum(dim=(0,1)), dim=-1) # [Φ]
            target_uniform_distribution = torch.full_like(aggr_att_score, fill_value = 1/args.mem_num) # [Φ]
            loss4 = F.kl_div(aggr_att_score, target_uniform_distribution, reduction='batchmean')
            loss = loss1 + args.lamb_cont * loss2 + args.lamb_cons * loss3 + args.lamb_kl * loss4

            val_loss.update(loss.item(), x.shape[0])
            val_mae_loss.update(loss1.item(), x.shape[0])
            val_cont_loss.update(loss2.item(), x.shape[0])
            val_cons_loss.update(loss3.item(), x.shape[0])
            val_kl_loss.update(loss4.item(), x.shape[0])
            val_num_slots.update(activated_memory_slots, 1)
            val_sparsity.update(sparsity, 1)

    return val_loss.avg, val_mae_loss.avg, val_cont_loss.avg, val_cons_loss.avg, val_kl_loss.avg, val_num_slots.avg, val_sparsity.avg
    

def test(args, model, dataloader):

    device = args.device

    y_pred_list = []
    y_true_list = []
    labels_list = []
    att_score_list = []
    test_num_slots = AverageMeter()
    test_sparsity = AverageMeter()

    model.eval()
    for x, labels in dataloader:
        x = x.unsqueeze(-1).to(device)
        labels = labels.to(device) 

        with torch.no_grad():
            output, m_t, query, pos, neg, att_score, adjs = model(x)
            activated_memory_slots = len(torch.unique(torch.argmax(att_score, dim=-1)))
            s1 = 1 - (torch.sum(adjs[0]) / args.num_nodes**2)
            s2 = 1 - (torch.sum(adjs[1]) / args.num_nodes**2)
            sparsity = (s1 + s2) / 2

            y_pred_list.append(output[:,-1,:,:].squeeze().detach().cpu())
            y_true_list.append(x[:,-1,:,:].squeeze().detach().cpu())
            labels_list.append(labels.squeeze().detach().cpu())
            att_score_list.append(att_score.detach().cpu()) 
            test_num_slots.update(activated_memory_slots, 1)
            test_sparsity.update(sparsity, 1)

    y_pred_np = torch.cat(y_pred_list, dim=0).numpy() 
    y_true_np = torch.cat(y_true_list, dim=0).numpy() 
    labels_np = torch.cat(labels_list, dim=0).numpy() 
    att_score_np = torch.cat(att_score_list, dim=0).numpy() 
    args.logger.info(f'Total {test_num_slots.avg:.3f} memory slots activated in average')
    args.logger.info(f'Average sparsity: {test_sparsity.avg:.3f}')

    return y_pred_np, y_true_np, labels_np, att_score_np
