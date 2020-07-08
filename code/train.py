import os
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['NUMEXPR_MAX_THREADS'] = '24'
import time
import math
import torch
import logging
import random
import argparse
import cate_loader
import cate_model
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings(action='ignore')


VERSION='v17'
DB_PATH=f'../../input/processed_{VERSION}'
MODEL_PATH=f'../../models/{VERSION}'
VOCAB_DIR=os.path.join(DB_PATH, 'vocab')


class CFG:
    learning_rate=1.0e-4
    batch_size=16
    num_workers=14
    print_freq=100
    test_freq=1
    start_epoch=0
    num_train_epochs=10
    warmup_steps=100
    max_grad_norm=10    
    weight_decay=0.01    
    dropout=0.2    
    hidden_size=512
    intermediate_size=256
    nlayers=2
    nheads=8
    seq_len=64
    n_b_cls = 57 + 1
    n_m_cls = 552 + 1
    n_s_cls = 3190 + 1
    n_d_cls = 404 + 1
    vocab_size = 32000
    img_feat_size = 2048
    type_vocab_size = 30
    df_path = os.path.join(DB_PATH, 'train.csv')
    h5_path = os.path.join(DB_PATH, 'train_img_feat.h5')


def main():    
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)    
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)    
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--wsteps", type=int, default=CFG.warmup_steps)
    parser.add_argument("--seed", type=int, default=7)        
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)    
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)    
    args = parser.parse_args()
    #print(args) 
    
    CFG.batch_size=args.batch_size
    CFG.num_train_epochs=args.nepochs
    CFG.seq_len=args.seq_len    
    CFG.warmup_steps=args.wsteps    
    CFG.learning_rate=args.lr
    CFG.dropout=args.dropout
    CFG.seed =  args.seed        
    CFG.nlayers =  args.nlayers    
    CFG.nheads =  args.nheads
    CFG.hidden_size =  args.hidden_size
    CFG.res_dir=f'res_dir_{args.k}'
    print(CFG.__dict__)    
    
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)    
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    
    print('loading ...')
    train_df = pd.read_csv(os.path.join(DB_PATH, 'train.csv'), dtype={'tokens':str})    
    train_df['img_idx'] = train_df.index
    if 1:
        train_df['unique_cateid'] = (train_df['bcateid'].astype('str') + train_df['mcateid'].astype('str') + 
                                                train_df['scateid'].astype('str') + train_df['dcateid'].astype('str'))
        folds = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
        train_idx, valid_idx = list(folds.split(train_df.values, train_df['unique_cateid']))[args.k]
        valid_df = train_df.iloc[valid_idx]
        train_df = train_df.iloc[train_idx]
    else:
        train_df, valid_df = train_test_split(train_df, test_size=10000)
    img_h5_path = os.path.join(DB_PATH, 'train_img_feat.h5')
    
    vocab = [line.split('\t')[0] for line in open(os.path.join(VOCAB_DIR, 'spm.vocab')).readlines()]
    token2id = dict([(w, i) for i, w in enumerate(vocab)])
    print('loading ... done')
        
    model = cate_model.CateClassifierl(CFG)
    if args.model != "":
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)        
        state_dict = checkpoint['state_dict']
        if args.resume:
            CFG.start_epoch = checkpoint['epoch']                
        model.load_state_dict(state_dict, strict=False)        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model, checkpoint['epoch']))        
    
    model.cuda()
    model._dropout = CFG.dropout
    print('model.dropout:', model._dropout)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameters: ', count_parameters(model))
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    train_db = cate_loader.CateDataset(train_df, img_h5_path, token2id, CFG.seq_len, CFG.type_vocab_size)
    valid_db = cate_loader.CateDataset(valid_df, img_h5_path, token2id, CFG.seq_len, CFG.type_vocab_size)
    
    train_loader = DataLoader(
        train_db, batch_size=CFG.batch_size, shuffle=True, drop_last=True,
        num_workers=CFG.num_workers, pin_memory=True)
    
    valid_loader = DataLoader(
        valid_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=False)
    
    num_train_optimization_steps = int(
        len(train_db) / CFG.batch_size) * (CFG.num_train_epochs)
    print('num_train_optimization_steps', num_train_optimization_steps)    
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                           lr=CFG.learning_rate,
                           weight_decay=CFG.weight_decay,                           
                           )

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
                                        num_training_steps=num_train_optimization_steps
                                     )
    print('use WarmupLinearSchedule ...')
    
    def get_lr():        
        return scheduler.get_lr()[0]
    
    if args.model != "":
        log_df = checkpoint['log']
        del checkpoint
    else:
        log_df = pd.DataFrame()     
    os.makedirs('log', exist_ok=True)
    
    curr_lr = get_lr()    
    print(f'initial learning rate:{curr_lr}')
            
    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):       
        
        def get_log_row_df(epoch, lr, train_res, valid_res):
            log_row = {'EPOCH':epoch, 'LR':lr,
                       'TRAIN_LOSS':train_res[0], 'TRAIN_OACC':train_res[1],
                       'TRAIN_BACC':train_res[2], 'TRAIN_MACC':train_res[3],
                       'TRAIN_SACC':train_res[4], 'TRAIN_DACC':train_res[5],
                       'VALID_LOSS':valid_res[0], 'VALID_OACC':valid_res[1],
                       'VALID_BACC':valid_res[2], 'VALID_MACC':valid_res[3],
                       'VALID_SACC':valid_res[4], 'VALID_DACC':valid_res[5],
                       }
            return pd.DataFrame(log_row, index=[0])             
            
        train_res = train(train_loader, model, optimizer, epoch, scheduler)
        valid_res = validate(valid_loader, model)
        curr_lr = get_lr()       
        print(f'set the learning_rate: {curr_lr}')      
        
        if epoch % CFG.test_freq == 0 and epoch >= 0:
            log_row_df = get_log_row_df(epoch, curr_lr, train_res, valid_res)
            log_df = log_df.append(log_row_df, sort=False)
            print(log_df.tail(20))
            
            curr_model_name = (f'b{CFG.batch_size}_h{CFG.hidden_size}_'
                               f'd{CFG.dropout}_l{CFG.nlayers}_hd{CFG.nheads}_'
                               f'ep{epoch}_s{CFG.seed}_k{args.k}.pt')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the cust_model it-self
                        
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'transformer',
                'state_dict': model_to_save.state_dict(),
                'log': log_df,
                },
                MODEL_PATH, curr_model_name,
            )
            
    print('done')


def calc_cate_acc(pred, label):
    b_pred, m_pred, s_pred, d_pred= pred    
    _, b_idx = b_pred.max(1)
    _, m_idx = m_pred.max(1)
    _, s_idx = s_pred.max(1)
    _, d_idx = d_pred.max(1)
        
    b_acc = (b_idx == label[:, 0]).sum().item() / (label[:, 0]>0).sum().item()
    m_acc = (m_idx == label[:, 1]).sum().item() / (label[:, 1]>0).sum().item()
            
    s_acc = (s_idx == label[:, 2]).sum().item() / ((label[:, 2]>0).sum().item()+1e-06)
    d_acc = (d_idx == label[:, 3]).sum().item() / ((label[:, 3]>0).sum().item()+1e-06)    
    o_acc = (b_acc + 1.2*m_acc + 1.3*s_acc + 1.4*d_acc)/4
    return o_acc, b_acc, m_acc, s_acc, d_acc


def train(train_loader, model, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    o_accuracies = AverageMeter() # overall
    b_accuracies = AverageMeter()
    m_accuracies = AverageMeter()
    s_accuracies = AverageMeter()
    d_accuracies = AverageMeter()
    
    sent_count = AverageMeter()
    
    # switch to train mode
    model.train()

    start = end = time.time()
    
    for step, (token_ids, token_mask, token_types, img_feat, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        token_ids, token_mask, token_types, img_feat, label = (token_ids.cuda(), 
                                                  token_mask.cuda(), token_types.cuda(), img_feat.cuda(), label.cuda())
                
        batch_size = token_ids.size(0)   
                
        # compute loss
        loss, pred = model(token_ids, token_mask, token_types, img_feat, label)
        loss = loss.mean()
                
        # record loss
        losses.update(loss.item(), batch_size)
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
              
        scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            o_acc, b_acc, m_acc, s_acc, d_acc = calc_cate_acc(pred, label)
            o_accuracies.update(o_acc, batch_size)
            b_accuracies.update(b_acc, batch_size)
            m_accuracies.update(m_acc, batch_size)
            s_accuracies.update(s_acc, batch_size)
            d_accuracies.update(d_acc, batch_size)
            
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                  'OAcc: {o_acc.val:.3f}({o_acc.avg:.3f}) '
                  'BAcc: {b_acc.val:.3f}({b_acc.avg:.3f}) '
                  'MAcc: {m_acc.val:.4f}({m_acc.avg:.3f}) '
                  'SAcc: {s_acc.val:.3f}({s_acc.avg:.3f}) '
                  'DAcc: {d_acc.val:.3f}({d_acc.avg:.3f}) '                  
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'sent/s {sent_s:.0f} '
                  .format(
                   epoch, step+1, len(train_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses,
                   o_acc=o_accuracies, b_acc=b_accuracies, m_acc=m_accuracies,
                   s_acc=s_accuracies, d_acc=d_accuracies,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],                   
                   sent_s=sent_count.avg/batch_time.avg
                   ))        
    return (losses.avg, o_accuracies.avg, b_accuracies.avg, m_accuracies.avg, 
            s_accuracies.avg, d_accuracies.avg) 


def validate(valid_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    o_accuracies = AverageMeter() # overall
    b_accuracies = AverageMeter()
    m_accuracies = AverageMeter()
    s_accuracies = AverageMeter()
    d_accuracies = AverageMeter()
    
    sent_count = AverageMeter()
    
    # switch to evaluation mode
    model.eval()

    start = end = time.time()
        
    for step, (token_ids, token_mask, token_types, img_feat, label) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        token_ids, token_mask, token_types, img_feat, label = (token_ids.cuda(), 
                                                  token_mask.cuda(), token_types.cuda(), img_feat.cuda(), label.cuda())
        
        batch_size = token_ids.size(0)
        
        with torch.no_grad():
            # compute loss
            loss, pred = model(token_ids, token_mask, token_types, img_feat, label)
            loss = loss.mean()
                
        # record loss
        losses.update(loss.item(), batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            o_acc, b_acc, m_acc, s_acc, d_acc = calc_cate_acc(pred, label)
            o_accuracies.update(o_acc, batch_size)
            b_accuracies.update(b_acc, batch_size)
            m_accuracies.update(m_acc, batch_size)
            s_accuracies.update(s_acc, batch_size)
            d_accuracies.update(d_acc, batch_size)
            
            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'OAcc: {o_acc.val:.3f}({o_acc.avg:.3f}) '
                  'BAcc: {b_acc.val:.3f}({b_acc.avg:.3f}) '
                  'MAcc: {m_acc.val:.4f}({m_acc.avg:.3f}) '
                  'SAcc: {s_acc.val:.3f}({s_acc.avg:.3f}) '
                  'DAcc: {d_acc.val:.3f}({d_acc.avg:.3f}) '
                  'sent/s {sent_s:.0f} '
                  .format(
                   step+1, len(valid_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses,
                   o_acc=o_accuracies, b_acc=b_accuracies, m_acc=m_accuracies,
                   s_acc=s_accuracies, d_acc=d_accuracies,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))        
    return (losses.avg, o_accuracies.avg, b_accuracies.avg, m_accuracies.avg, 
            s_accuracies.avg, d_accuracies.avg)


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()


def save_checkpoint(state, model_path, model_filename, is_best=False):
    print('saving cust_model ...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, os.path.join(model_path, model_filename))
    if is_best:
        torch.save(state, os.path.join(model_path, 'best_' + model_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def adjust_learning_rate(optimizer, epoch):  
    #lr  = CFG.learning_rate     
    lr = (CFG.lr_decay)**(epoch//10) * CFG.learning_rate    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    return lr




if __name__ == '__main__':
    main()
