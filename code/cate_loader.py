import torch
from torch.utils.data import Dataset
import h5py
import re


class CateDataset(Dataset):
    def __init__(self, df_data, img_h5_path, token2id, token_len=64, type_vocab_size=30,
                 inverse=False):
        
        self.tokens = df_data['tokens'].values
        self.img_indices = df_data['img_idx'].values
        self.img_h5_path = img_h5_path#h5_data['img_feat']
        self.token_len = token_len        
        self.labels = df_data[['bcateid', 'mcateid', 'scateid', 'dcateid']].values
        self.token2id = token2id 
        self.p = re.compile('▁[^▁]+')
        self.type_vocab_size = type_vocab_size
        self.inverse = inverse
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        
        tokens = self.tokens[idx]
        if not isinstance(tokens, str):
            tokens = ''
        
        tokens = self.p.findall(tokens)
        
        if self.inverse:
            tokens = tokens[::-1] 
        
        token_types = [type_id for type_id, word in enumerate(tokens) for _ in word.split()]
        tokens = " ".join(tokens)
        token_ids = [self.token2id[tok] if tok in self.token2id else 0 for tok in tokens.split()]
                
        if len(token_ids) > self.token_len:
            token_ids = token_ids[:self.token_len]         
            token_types = token_types[:self.token_len]
                 
        token_mask = [1] * len(token_ids)
        token_pad = [0] * (self.token_len - len(token_ids))
        token_ids += token_pad
        token_mask += token_pad
        token_types += token_pad
        with h5py.File(self.img_h5_path, 'r') as img_feats:
            img_feat = img_feats['img_feat'][self.img_indices[idx]]
        
        token_ids = torch.LongTensor(token_ids)
        token_mask = torch.LongTensor(token_mask)
        token_types = torch.LongTensor(token_types)
        token_types[token_types >= self.type_vocab_size] = self.type_vocab_size-1 
        img_feat = torch.FloatTensor(img_feat)
        
        label = self.labels[idx]
        label = torch.LongTensor(label)
        
        return token_ids, token_mask, token_types, img_feat, label
    
    def __len__(self):
        return len(self.tokens)

    
DB_PATH='../../input/processed_v17'
VOCAB_DIR='../../input/processed_v17/vocab'

def main():
    train_df = pd.read_csv(os.path.join(DB_PATH, 'dev.csv'))
    train_df['img_idx'] = train_df.index 
    train_df['unique_cateid'] = (train_df['bcateid'].astype('str') + train_df['mcateid'].astype('str') + 
                                            train_df['scateid'].astype('str') + train_df['dcateid'].astype('str'))
    img_feat_path = os.path.join(DB_PATH, 'dev_img_feat.h5')    
    vocab = [line.split('\t')[0] for line in open(os.path.join(VOCAB_DIR, 'spm.vocab')).readlines()]
    token2id = dict([(w, i) for i, w in enumerate(vocab)])
    
    train_db = CateDataset(train_df, img_feat_path, token2id)
    
    train_loader = DataLoader(
        train_db, batch_size=1024, shuffle=False,
        num_workers=16, pin_memory=True)
    
    sum_img_feat = 0
    for token_ids, token_mask, token_types, img_feat, label in tqdm(train_loader):
        sum_img_feat += img_feat.sum().item()
        a = 0 
    print(sum_img_feat)