from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import *
from test import *
from evaluate import *


def train(args, model, train_dataloader, val_dataloader, test_dataloader):
    device = args.device
    contrastive_loss = nn.TripletMarginLoss(margin=1.0)
    consistency_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)

    best_loss = float('inf')
    early_stop_win = args.patience
    stop_improve_count = 0
    batches_seen = 0

    for i_epoch in range(args.epochs):

        train_loss = AverageMeter()
        train_mae_loss = AverageMeter()
        train_cont_loss = AverageMeter()
        train_cons_loss = AverageMeter()
        train_kl_loss = AverageMeter()  
        train_num_slots = AverageMeter() 
        train_sparsity = AverageMeter()
        
        model.train()
        for x, _ in tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch ({i_epoch+1}/{args.epochs})'):            
            optimizer.zero_grad()
            x = x.unsqueeze(-1).to(device)

            ## Forward
            output, m_t, query, pos, neg, att_score, adjs = model(x, batches_seen)
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

            train_loss.update(loss.item(), x.shape[0])
            train_mae_loss.update(loss1.item(), x.shape[0])
            train_cont_loss.update(loss2.item(), x.shape[0])
            train_cons_loss.update(loss3.item(), x.shape[0])
            train_kl_loss.update(loss4.item(), x.shape[0])
            train_num_slots.update(activated_memory_slots, 1)
            train_sparsity.update(sparsity, 1)

            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        
        lr_scheduler.step()

        val_loss, val_mae_loss, val_cont_loss, val_cons_loss, val_kl_loss, val_num_slots, val_sparsity = validation(args, model, val_dataloader)

        args.logger.info(f'Epoch ({i_epoch+1}/{args.epochs}) | Train loss {train_loss.avg:.4f}({train_num_slots.avg:.3f},{train_sparsity.avg:.3f}) / Val loss {val_loss:.4f}({val_num_slots:.3f},{val_sparsity:.3f})')
        if args.wandb:
            wandb.log({"train_loss":train_loss.avg, "train_mae_loss":train_mae_loss.avg, "train_cont_loss":train_cont_loss.avg, "train_cons_loss":train_cons_loss.avg, "train_kl_loss":train_kl_loss.avg, "train_num_slots":train_num_slots.avg, "train_sparsity":train_sparsity.avg,
                       "val_loss":val_loss, "val_mae_loss":val_mae_loss, "val_cont_loss":val_cont_loss, "val_cons_loss":val_cons_loss, "val_kl_loss":val_kl_loss, "val_num_slots":val_num_slots, "val_sparsity":val_sparsity})

        if val_loss < best_loss:
            args.logger.info(f'Lower val loss, saving the model.. {best_loss:.5f} → {val_loss:.5f}')
            torch.save(model.state_dict(), f'{args.log_dir}/best.pth')
            best_loss = val_loss
            stop_improve_count = 0
        else:
            stop_improve_count += 1

        if stop_improve_count >= early_stop_win:
            args.logger.info(f'Early stop triggered. Aborting..')
            break

    args.logger.info('Finished Training. Saved the best model.')