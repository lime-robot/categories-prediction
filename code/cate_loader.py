
import torch # 파이토치 패키지 임포트
from torch.utils.data import Dataset # Dataset 클래스 임포트
import h5py # h5py 패키지 임포트
import re # 정규식표현식 모듈 임포트 


class CateDataset(Dataset):
    """데이터셋에서 학습에 필요한 형태로 변환된 샘플 하나를 반환
    """
    def __init__(self, df_data, img_h5_path, token2id, tokens_max_len=64, type_vocab_size=30):
        """        
        매개변수
        df_data: 상품타이틀, 카테고리 등의 정보를 가지는 데이터프레임
        img_h5_path: img_feat가 저장돼 있는 h5 파일의 경로
        token2id: token을 token_id로 변환하기 위한 맵핑 정보를 가진 딕셔너리
        tokens_max_len: tokens의 최대 길이. 상품명의 tokens가 이 이상이면 잘라서 버림
        type_vocab_size: 타입 사전의 크기
        """        
        self.tokens = df_data['tokens'].values # 전처리된 상품명
        self.img_indices = df_data['img_idx'].values # h5의 이미지 인덱스
        self.img_h5_path = img_h5_path 
        self.tokens_max_len = tokens_max_len        
        self.labels = df_data[['bcateid', 'mcateid', 'scateid', 'dcateid']].values
        self.token2id = token2id 
        self.p = re.compile('▁[^▁]+') # ▁기호를 기준으로 나누기 위한 컴파일된 정규식
        self.type_vocab_size = type_vocab_size
        
    def __getitem__(self, idx):
        """
        데이터셋에서 idx에 대응되는 샘플을 변환하여 반환        
        """
        if idx >= len(self):
            raise StopIteration
        
        # idx에 해당하는 상품명 가져오기. 상품명은 문자열로 저장돼 있음
        tokens = self.tokens[idx]
        if not isinstance(tokens, str):
            tokens = ''
        
        # 상품명을 ▁기호를 기준으로 분리하여 파이썬 리스트로 저장
        # "▁직소퍼즐 ▁1000 조각 ▁바다 거북 의 ▁여행 ▁pl 12 75" =>
        # ["▁직소퍼즐", "▁1000 조각", "▁바다 거북 의", "▁여행", "▁pl 12 75"]
        tokens = self.p.findall(tokens)
        
        # ▁ 기호 별 토큰타입 인덱스 부여
        # ["▁직소퍼즐", "▁1000 조각", "▁바다 거북 의", "▁여행", "▁pl 12 75"] =>
        # [     0     ,     1    1  ,    2     2  2 ,     3   ,   4  4   4 ]
        token_types = [type_id for type_id, word in enumerate(tokens) for _ in word.split()]       
        tokens = " ".join(tokens) # ▁기호로 분리되기 전의 원래의 tokens으로 되돌림

        # 토큰을 토큰에 대응되는 인덱스로 변환
        # "▁직소퍼즐 ▁1000 조각 ▁바다 거북 의 ▁여행 ▁pl 12 75" =>
        # [2291, 784, 2179, 3540, 17334, 30827, 1114, 282, 163, 444]
        # "▁직소퍼즐" => 2291
        # "▁1000" => 784
        # "조각" => 2179
        # ...
        token_ids = [self.token2id[tok] if tok in self.token2id else 0 for tok in tokens.split()]
        
        # token_ids의 길이가 max_len보다 길면 잘라서 버림
        if len(token_ids) > self.tokens_max_len:
            token_ids = token_ids[:self.tokens_max_len]      
            token_types = token_types[:self.tokens_max_len]
        
        # token_ids의 길이가 max_len보다 짧으면 짧은만큼 PAD값 0 값으로 채워넣음
        # token_ids 중 값이 있는 곳은 1, 그 외는 0으로 채운 token_mask 생성
        token_mask = [1] * len(token_ids)
        token_pad = [0] * (self.tokens_max_len - len(token_ids))
        token_ids += token_pad
        token_mask += token_pad
        token_types += token_pad # max_len 보다 짧은만큼 PAD 추가

        # h5파일에서 이미지 인덱스에 해당하는 img_feat를 가져옴
        # 파이토치의 데이터로더에 의해 동시 h5파일에 동시접근이 발생해도
        # 안정적으로 img_feat를 가져오려면 아래처럼 매번 h5py.File 호출필요
        with h5py.File(self.img_h5_path, 'r') as img_feats:
            img_feat = img_feats['img_feat'][self.img_indices[idx]]
        
        # 넘파이(numpy)나 파이썬 자료형을 파이토치의 자료형으로 변환
        token_ids = torch.LongTensor(token_ids)
        token_mask = torch.LongTensor(token_mask)
        token_types = torch.LongTensor(token_types)
        
        # token_types의 타입 인덱스의 숫자 크기가 type_vocab_size 보다 작도록 바꿈
        token_types[token_types >= self.type_vocab_size] = self.type_vocab_size-1 
        img_feat = torch.FloatTensor(img_feat)
        
        # 대/중/소/세 라벨 준비
        label = self.labels[idx]
        label = torch.LongTensor(label)
        
        # 크게 3가지 텍스트 입력, 이미지 입력, 라벨을 반환한다.
        return token_ids, token_mask, token_types, img_feat, label
    
    def __len__(self):
        """
          tokens의 개수를 반환한다. 즉, 상품명 문장의 개수를 반환한다.
        """
        return len(self.tokens)
