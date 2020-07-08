import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers.modeling_bert import BertConfig, BertEncoder, BertModel


class CateClassifierl(nn.Module):
    def __init__(self, cfg):
        super(CateClassifierl, self).__init__()
        self.cfg = cfg
                
        self.bert_cfg = BertConfig( 
            cfg.vocab_size, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.intermediate_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
            max_position_embeddings=cfg.seq_len,
            type_vocab_size=cfg.type_vocab_size,
        )
        self.text_emb = BertModel(self.bert_cfg)
        self.img_emb = nn.Linear(cfg.img_feat_size, cfg.hidden_size)
                
        def get_cls(target_size=1):
            return nn.Sequential(
                nn.Linear(cfg.hidden_size*2, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, target_size),
            )        
               
        self.b_cls = get_cls(cfg.n_b_cls)
        self.m_cls = get_cls(cfg.n_m_cls)
        self.s_cls = get_cls(cfg.n_s_cls)
        self.d_cls = get_cls(cfg.n_d_cls)
    
    def forward(self, token_ids, token_mask, token_types, img_feat, label=None):
        text_emb = self.text_emb(token_ids, token_mask, token_types)[0]
        text_emb = text_emb[:, 0]
        img_emb = self.img_emb(img_feat)
        
        comb_emb = torch.cat([text_emb, img_emb], 1)
        b_pred = self.b_cls(comb_emb)
        m_pred = self.m_cls(comb_emb)
        s_pred = self.s_cls(comb_emb)
        d_pred = self.d_cls(comb_emb)
        
        if label is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=-1)
            b_label, m_label, s_label, d_label = label.split(1, 1)
            b_loss = loss_func(b_pred, b_label.view(-1))
            m_loss = loss_func(m_pred, m_label.view(-1))
            s_loss = loss_func(s_pred, s_label.view(-1))
            d_loss = loss_func(d_pred, d_label.view(-1))                        
            loss = (b_loss + 1.2*m_loss + 1.3*s_loss + 1.4*d_loss)/4    
        else:
            loss = b_pred.new(1).fill_(0)      
        
        return loss, [b_pred, m_pred, s_pred, d_pred]

