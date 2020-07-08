import os
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['NUMEXPR_MAX_THREADS'] = '24'
import math
import glob
import torch
import cate_loader
import cate_model
import time
import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings(action='ignore')

import argparse
import logging


VERSION='v17'
DB_PATH=f'../../input/processed_{VERSION}'
MODEL_PATH=f'../../models/{VERSION}'
VOCAB_DIR=os.path.join(DB_PATH, 'vocab')


class CFG:
    learning_rate=1.0e-3
    batch_size=16
    num_workers=14
    print_freq=100
    test_freq=1
    start_epoch=0
    num_train_epochs=5    
    warmup_steps=100
    max_grad_norm=10
    gradient_accumulation_steps=1
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
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)   
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)    
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)    
    args = parser.parse_args()
    print(args) 
    
    CFG.batch_size=args.batch_size
    CFG.dropout=args.dropout
    CFG.seed =  args.seed        
    CFG.nlayers =  args.nlayers    
    CFG.nheads =  args.nheads
    CFG.hidden_size =  args.hidden_size
    CFG.seq_len =  args.seq_len
    CFG.res_dir=f'res_dir_{args.k}'
    print(CFG.__dict__)    
    
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)    
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    
    print('loading ...')
    valid_df = pd.read_csv(os.path.join(DB_PATH, 'dev.csv'), dtype={'tokens':str})     
    valid_df['img_idx'] = valid_df.index
    img_h5_path = os.path.join(DB_PATH, 'dev_img_feat.h5')
    
    vocab = [line.split('\t')[0] for line in open(os.path.join(VOCAB_DIR, 'spm.vocab')).readlines()]
    token2id = dict([(w, i) for i, w in enumerate(vocab)])    
    print('loading ... done')
        
    model_list = []
    model_path_list = glob.glob(os.path.join(args.model_dir, '*.pt'))
    for model_path in model_path_list:
        model = cate_model.CateClassifierl(CFG)
        if model_path != "":
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)        
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict, strict=True)        
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        model.cuda()
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model_list.append(model)
    if len(model_list) == 0:
        print('Please check the model directory.')
        return
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameters: ', count_parameters(model_list[0]))    
    
    valid_db = cate_loader.CateDataset(valid_df, img_h5_path, token2id, CFG.seq_len, 
                                       CFG.type_vocab_size)
    
    valid_loader = DataLoader(
        valid_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True)    
    
    pred_idx = validate(valid_loader, model_list)
    
    cate_cols = ['bcateid', 'mcateid', 'scateid', 'dcateid'] 
    valid_df[cate_cols] = pred_idx    
    valid_df[['pid'] + cate_cols].to_csv(f'dev.tsv', sep='\t', header=False, index=False)
            
    print('done')


def get_pred_idx(pred):
    b_pred, m_pred, s_pred, d_pred= pred    
    _, b_idx = b_pred.max(1)
    _, m_idx = m_pred.max(1)
    _, s_idx = s_pred.max(1)
    _, d_idx = d_pred.max(1)
    pred_idx = torch.stack([b_idx, m_idx, s_idx, d_idx], 1)    
    return pred_idx


def blend_pred_list(pred_list, t=0.3):
    b_pred, m_pred, s_pred, d_pred = 0, 0, 0, 0
    for pred in pred_list:
        b_pred += torch.softmax(pred[0], 1) ** t
        m_pred += torch.softmax(pred[1], 1) ** t
        s_pred += torch.softmax(pred[2], 1) ** t
        d_pred += torch.softmax(pred[3], 1) ** t
    b_pred /= len(pred_list)
    m_pred /= len(pred_list)
    s_pred /= len(pred_list)
    d_pred /= len(pred_list)
    pred = [b_pred, m_pred, s_pred, d_pred]    
    return pred


def validate(valid_loader, model_list):
    batch_time = AverageMeter()
    data_time = AverageMeter()    
    sent_count = AverageMeter()
    
    # switch to evaluation mode
    for model in model_list:
        model.eval()

    start = end = time.time()
    
    pred_idx_list = []
    
    sum_img_feat = 0
    sum_token_ids = 0
    for step, (token_ids, token_mask, token_types, img_feat, _) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        token_ids, token_mask, token_types, img_feat = (token_ids.cuda(), 
                                                  token_mask.cuda(), token_types.cuda(), img_feat.cuda())
        
        batch_size = token_ids.size(0)
        
        sum_img_feat += img_feat.sum().item()
        sum_token_ids += token_ids.sum().item()
        
        with torch.no_grad():
            # compute loss
            pred_list = []
            for model in model_list:
                _, pred = model(token_ids, token_mask, token_types, img_feat)
                pred_list.append(pred)
            
            pred = blend_pred_list(pred_list)                
            pred_idx = get_pred_idx(pred)
            pred_idx_list.append(pred_idx.cpu())
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '                  
                  'sent/s {sent_s:.0f} '
                  .format(
                   step+1, len(valid_loader), batch_time=batch_time,                   
                   data_time=data_time,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))
    pred_idx = torch.cat(pred_idx_list).numpy()
    
    print(sum_img_feat, sum_token_ids)
    
    return pred_idx


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
