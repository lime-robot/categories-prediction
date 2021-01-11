import os
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['NUMEXPR_MAX_THREADS'] = '24'
import math
import glob
import json
import torch
import cate_dataset
import cate_model
import time
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings(action='ignore')
import argparse


# 전처리된 데이터가 저장된 디렉터리
DB_DIR = '../input/processed'

# 토큰을 인덱스로 치환할 때 사용될 사전 파일이 저장된 디렉터리 
VOCAB_DIR = os.path.join(DB_DIR, 'vocab')

# 학습된 모델의 파라미터가 저장될 디렉터리
MODEL_DIR = '../model'

# 제출할 예측결과가 저장될 디렉터리
SUBMISSION_DIR = '../submission'


# 미리 정의된 설정 값
class CFG:    
    batch_size=1024 # 배치 사이즈
    num_workers=4 # 워커의 개수
    print_freq=100 # 결과 출력 빈도    
    warmup_steps=100 # lr을 서서히 증가시킬 step 수        
    hidden_size=512 # 은닉 크기
    dropout=0.2 # dropout 확률
    intermediate_size=256 # TRANSFORMER셀의 intermediate 크기
    nlayers=2 # BERT의 층수
    nheads=8 # BERT의 head 개수
    seq_len=64 # 토큰의 최대 길이
    n_b_cls = 57 + 1 # 대카테고리 개수
    n_m_cls = 552 + 1 # 중카테고리 개수
    n_s_cls = 3190 + 1 # 소카테고리 개수
    n_d_cls = 404 + 1 # 세카테고리 개수
    vocab_size = 32000 # 토큰의 유니크 인덱스 개수
    img_feat_size = 2048 # 이미지 피처 벡터의 크기
    type_vocab_size = 30 # 타입의 유니크 인덱스 개수
    csv_path = os.path.join(DB_DIR, 'dev.csv') # 전처리 돼 저장된 dev 데이터셋    
    h5_path = os.path.join(DB_DIR, 'dev_img_feat.h5')

    

def main():
    # 명령행에서 받을 키워드 인자를 설정합니다.
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)    
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)   
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--nworkers", type=int, default=CFG.num_workers)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)    
    parser.add_argument("--k", type=int, default=0)    
    args = parser.parse_args()
    print(args) 
    
    CFG.batch_size=args.batch_size    
    CFG.seed =  args.seed        
    CFG.nlayers =  args.nlayers    
    CFG.nheads =  args.nheads
    CFG.hidden_size =  args.hidden_size
    CFG.seq_len =  args.seq_len
    CFG.num_workers=args.nworkers
    CFG.res_dir=f'res_dir_{args.k}'
    print(CFG.__dict__)    
    
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 함
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)    
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    
    # 전처리된 데이터를 읽어옵니다.
    print('loading ...')
    dev_df = pd.read_csv(CFG.csv_path, dtype={'tokens':str})     
    dev_df['img_idx'] = dev_df.index
    img_h5_path = CFG.h5_path
    
    vocab = [line.split('\t')[0] for line in open(os.path.join(VOCAB_DIR, 'spm.vocab'), encoding='utf-8').readlines()]
    token2id = dict([(w, i) for i, w in enumerate(vocab)])    
    print('loading ... done')
        
    # 찾아진 모델 파일의 개수만큼 모델을 만들어서 파이썬 리스트에 추가함
    model_list = []
    # args.model_dir에 있는 확장자 .pt를 가지는 모든 모델 파일의 경로를 읽음
    model_path_list = glob.glob(os.path.join(args.model_dir, '*.pt'))
    # 모델 경로 개수만큼 모델을 생성하여 파이썬 리스트에 추가함
    for model_path in model_path_list:
        model = cate_model.CateClassifier(CFG)
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
    
    # 모델의 파라미터 수를 출력합니다.
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameters: ', count_parameters(model_list[0]))    
    
    # 모델의 입력에 적합한 형태의 샘플을 가져오는 CateDataset의 인스턴스를 만듦
    dev_db = cate_dataset.CateDataset(dev_df, img_h5_path, token2id, CFG.seq_len, 
                                       CFG.type_vocab_size)
    
    # 여러 개의 워커로 빠르게 배치(미니배치)를 생성하도록 DataLoader로 
    # CateDataset 인스턴스를 감싸 줌
    dev_loader = DataLoader(
        dev_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True)    
    
    # dev 데이터셋의 모든 상품명에 대해 예측된 카테고리 인덱스를 반환
    pred_idx = inference(dev_loader, model_list)
    
    # dev 데이터셋의 상품ID별 예측된 카테고리를 붙여서 제출 파일을 생성하여 저장
    cate_cols = ['bcateid', 'mcateid', 'scateid', 'dcateid'] 
    dev_df[cate_cols] = pred_idx
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, 'dev.tsv')
    dev_df[['pid'] + cate_cols].to_csv(submission_path, sep='\t', header=False, index=False)
            
    print('done')


def inference(dev_loader, model_list):
    """
    dev 데이터셋의 모든 상품명에 대해 여러 모델들의 예측한 결과를 앙상블 하여 정확도가 개선된
    카테고리 인덱스를 반환
    
    매개변수
    dev_loader: dev 데이터셋에서 배치(미니배치) 불러옴
    model_list: args.model_dir에서 불러온 모델 리스트 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()    
    sent_count = AverageMeter()
    
    # 모딜 리스트의 모든 모델을 평가(evaluation) 모드로 작동하게 함
    for model in model_list:
        model.eval()

    start = end = time.time()
    
    # 배치별 예측한 대/중/소/세 카테고리의 인덱스를 리스트로 가짐
    pred_idx_list = []
    
    # dev_loader에서 반복해서 배치 데이터를 받음
    # CateDataset의 __getitem__() 함수의 반환 값과 동일한 변수 반환
    for step, (token_ids, token_mask, token_types, img_feat, _) in enumerate(dev_loader):
        # 데이터 로딩 시간 기록
        data_time.update(time.time() - end)
        
        # 배치 데이터의 위치를 CPU메모리에서 GPU메모리로 이동
        token_ids, token_mask, token_types, img_feat = (
            token_ids.cuda(), token_mask.cuda(), token_types.cuda(), img_feat.cuda())
        
        batch_size = token_ids.size(0)
        
        # with문 내에서는 그래디언트 계산을 하지 않도록 함
        with torch.no_grad():
            pred_list = []
            # model 별 예측치를 pred_list에 추가합니다.
            for model in model_list:
                _, pred = model(token_ids, token_mask, token_types, img_feat)
                pred_list.append(pred)
            
            # 예측치 리스트를 앙상블 하여 하나의 예측치로 만듦
            pred = ensemble(pred_list)
            # 예측치에서 카테고리별 인덱스를 가져옴
            pred_idx = get_pred_idx(pred)
            # 현재 배치(미니배치)에서 얻어진 카테고리별 인덱스를 pred_idx_list에 추가
            pred_idx_list.append(pred_idx.cpu())
            
        # 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(dev_loader)-1):
            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '                  
                  'sent/s {sent_s:.0f} '
                  .format(
                   step+1, len(dev_loader), batch_time=batch_time,                   
                   data_time=data_time,
                   remain=timeSince(start, float(step+1)/len(dev_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))
    
    # 배치별로 얻어진 카테고리 인덱스 리스트를 직렬연결하여 하나의 카테고리 인덱스로 변환
    pred_idx = torch.cat(pred_idx_list).numpy()
    return pred_idx

# 예측치의 각 카테고리 별로 가장 큰 값을 가지는 인덱스를 반환함
def get_pred_idx(pred):
    b_pred, m_pred, s_pred, d_pred= pred # 대/중/소/세 예측치로 분리
    _, b_idx = b_pred.max(1) # 대카테고리 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    _, m_idx = m_pred.max(1) # 중카테고리 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    _, s_idx = s_pred.max(1) # 소카테고리 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    _, d_idx = d_pred.max(1) # 세카테고리 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    
    # 대/중/소/세 인덱스 반환
    pred_idx = torch.stack([b_idx, m_idx, s_idx, d_idx], 1)    
    return pred_idx


# 예측된 대/중/소/세 결과들을 앙상블함
# 앙상블 방법으로 간단히 산술 평균을 사용
def ensemble(pred_list):
    b_pred, m_pred, s_pred, d_pred = 0, 0, 0, 0    
    for pred in pred_list:
        # softmax를 적용해 대/중/소/세 각 카테고리별 모든 클래스의 합이 1이 되도록 정규화
        # 참고로 정규화된 pred[0]은 대카테고리의 클래스별 확률값을 가지는 확률분포 함수라 볼 수 있음
        b_pred += torch.softmax(pred[0], 1)
        m_pred += torch.softmax(pred[1], 1)
        s_pred += torch.softmax(pred[2], 1)
        d_pred += torch.softmax(pred[3], 1)
    b_pred /= len(pred_list)    # 모델별 '대카테고리의 정규화된 예측값'들의 평균 계산
    m_pred /= len(pred_list)   # 모델별 '중카테고리의 정규화된 예측값'들의 평균 계산
    s_pred /= len(pred_list)    # 모델별 '소카테고리의 정규화된 예측값'들의 평균 계산
    d_pred /= len(pred_list)    # 모델별 '세카테고리의 정규화된 예측값'들의 평균 계산  
    
    # 앙상블 결과 반환 
    pred = [b_pred, m_pred, s_pred, d_pred]    
    return pred


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


if __name__ == '__main__':
    main()
