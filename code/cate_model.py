
import torch # 파이토치 패키지 임포트
import torch.nn as nn # 자주 사용하는 torch.nn패키지를 별칭 nn으로 명명
# 허깅페이스의 트랜스포머 패키지에서 BertConfig, BertModel 클래스 임포트
from transformers.modeling_bert import BertConfig, BertModel


class CateClassifier(nn.Module):
    """상품정보를 받아서 대/중/소/세 카테고리를 예측하는 모델    
    """
    def __init__(self, cfg):        
        """
        매개변수
        cfg: hidden_size, nlayers 등 설정값을 가지고 있는 변수
        """
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
        # 텍스트 인코더로 버트모델 사용
        self.text_encoder = BertModel(self.bert_cfg)
        # 이미지 인코더로 선형모델 사용(대회에서 이미지가 아닌 img_feat를 제공)
        self.img_encoder = nn.Linear(cfg.img_feat_size, cfg.hidden_size)
                
        # 분류기(Classifier) 생성기
        def get_cls(target_size=1):
            return nn.Sequential(
                nn.Linear(cfg.hidden_size*2, target_size),
                #nn.Linear(cfg.hidden_size*2, cfg.hidden_size),
                #nn.LayerNorm(cfg.hidden_size),
                #nn.Dropout(cfg.dropout),
                #nn.ReLU(),
                #nn.Linear(cfg.hidden_size, target_size),
            )        
          
        # 대 카테고리 분류기
        self.b_cls = get_cls(cfg.n_b_cls)
        # 중 카테고리 분류기
        self.m_cls = get_cls(cfg.n_m_cls)
        # 소 카테고리 분류기
        self.s_cls = get_cls(cfg.n_s_cls)
        # 세 카테고리 분류기
        self.d_cls = get_cls(cfg.n_d_cls)
    
    def forward(self, token_ids, token_mask, token_types, img_feat, label=None):
        """        
        매개변수
        token_ids: 전처리된 상품명을 인덱스로 변환하여 token_ids를 만들었음
        token_mask: 실제 token_ids의 개수만큼은 1, 나머지는 0으로 채움
        token_types: ▁ 문자를 기준으로 서로 다른 타입의 토큰임을 타입 인덱스로 저장
        img_feat: resnet50으로 인코딩된 이미지 피처
        label: 정답 대/중/소/세 카테고리
        """

        # 전처리된 상품명을 하나의 텍스트벡터(text_vec)로 변환
        # 반환 튜플(시퀀스 아웃풋, 풀드 아웃풋) 중 시퀀스 아웃풋만 참조 
        text_output = self.text_encoder(token_ids, token_mask, token_type_ids=token_types)[0]
        
        # 시퀀스 중 첫 타임스탭의 hidden state만 사용. 
        text_vec = text_output[:, 0]
        
        # img_feat를 텍스트벡터와 결합하기 앞서 선형변환 적용
        img_vec = self.img_encoder(img_feat)
        
        # 이미지벡터와 텍스트벡터를 직렬연결(concatenate)하여 결합벡터 생성
        comb_vec = torch.cat([text_vec, img_vec], 1)
        
        # 결합된 벡터로 대카테고리 확률분포 예측
        b_pred = self.b_cls(comb_vec)
        # 결합된 벡터로 중카테고리 확률분포 예측
        m_pred = self.m_cls(comb_vec)
        # 결합된 벡터로 소카테고리 확률분포 예측
        s_pred = self.s_cls(comb_vec)
        # 결합된 벡터로 세카테고리 확률분포 예측
        d_pred = self.d_cls(comb_vec)
        
        # 데이터 패러럴 학습에서 GPU 메모리를 효율적으로 사용하기 위해 
        # loss를 모델 내에서 계산함.
        if label is not None:
            # 손실(loss) 함수로 CrossEntropyLoss를 사용
            # label의 값이 -1을 가지는 샘플은 loss계산에 사용 안 함
            loss_func = nn.CrossEntropyLoss(ignore_index=-1)
            # label은 batch_size x 4를 (batch_size x 1) 4개로 만듦
            b_label, m_label, s_label, d_label = label.split(1, 1)
            # 대카테고리의 예측된 확률분포와 정답확률 분포의 차이를 손실로 반환
            b_loss = loss_func(b_pred, b_label.view(-1))
            # 중카테고리의 예측된 확률분포와 정답확률 분포의 차이를 손실로 반환
            m_loss = loss_func(m_pred, m_label.view(-1))
            # 소카테고리의 예측된 확률분포와 정답확률 분포의 차이를 손실로 반환
            s_loss = loss_func(s_pred, s_label.view(-1))
            # 세카테고리의 예측된 확률분포와 정답확률 분포의 차이를 손실로 반환
            d_loss = loss_func(d_pred, d_label.view(-1))
            # 대/중/소/세 손실의 평균을 낼 때 실제 대회 평가방법과 일치하도록 함
            loss = (b_loss + 1.2*m_loss + 1.3*s_loss + 1.4*d_loss)/4    
        else: # label이 없으면 loss로 0을 반환
            loss = b_pred.new(1).fill_(0)      
        
        # 최종 계산된 손실과 예측된 대/중/소/세 각 확률분포를 반환
        return loss, [b_pred, m_pred, s_pred, d_pred]

