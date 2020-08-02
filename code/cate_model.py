import torch # 파이토치 모듈 임포트
import torch.nn as nn # 자주 사용하는 torch.nn을 별칭 nn으로 명명
# 허깅페이스의 트랜스포머에서 BertConfig, BertModel 클래스 임포트
from transformers.modeling_bert import BertConfig, BertModel

# 파이토치로부터 커스텀 모델을 만들기 위해 nn.Module을 상속 받음
class CateClassifier(nn.Module):
    def __init__(self, cfg):
        super(CateClassifier, self).__init__()
        # 글로벌 설정값을 멤버 변수로 저장
        self.cfg = cfg
        # 버트모델의 설정값을 멤버 변수로 저장
        self.bert_cfg = BertConfig( 
            cfg.vocab_size, # 사전 크기
            hidden_size=cfg.hidden_size, # 히든 크기
            num_hidden_layers=cfg.nlayers, # 레이어 층 수
            num_attention_heads=cfg.nheads, # 어텐션 헤드의 수
            intermediate_size=cfg.intermediate_size, # 인터미디어트 크기
            hidden_dropout_prob=cfg.dropout, # 히든 드롭아웃 확률 값
            attention_probs_dropout_prob=cfg.dropout, # 어텐션 드롭아웃 확률 값 
            max_position_embeddings=cfg.seq_len, # 포지션 임베딩의 최대 길이
            type_vocab_size=cfg.type_vocab_size, # 타입 사전 크기
        )
        self.text_encoder = BertModel(self.bert_cfg)
        self.img_encoder = nn.Linear(cfg.img_feat_size, cfg.hidden_size)
                
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
        text_output = self.text_encoder(token_ids, token_mask, token_types)[0]
        text_vec = text_output[:, 0]
        img_vec = self.img_encoder(img_feat)
        
        comb_vec = torch.cat([text_vec, img_vec], 1)
        b_pred = self.b_cls(comb_vec)
        m_pred = self.m_cls(comb_vec)
        s_pred = self.s_cls(comb_vec)
        d_pred = self.d_cls(comb_vec)
        
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

