from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle
import os

class CustomDataset(Dataset):
    def __init__(self,path_dump, path_fea, input_size, zero_class):
        with open(os.path.join(path_dump), 'rb') as f :
            self.list_data = pickle.load(f)
        self.list_data = self.list_data['fea_list']
        self.path_fea = path_fea
        self.input_size = input_size
        self.zero_class = zero_class

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        file = self.list_data[idx]['file']
        num = self.list_data[idx]['num']
        with open(os.path.join(self.path_fea,'%s.pickle'%(file)),'rb') as f :
            fea = pickle.load(f)
        
        feature = [fea['mfcc'][:,i] for i in num]
        feature = np.array(feature)
        feature = np.transpose(feature)
        feature = feature[np.newaxis,:,:]
        label = [fea['label'][i] for i in num]
        if not self.zero_class :
            for i in range(len(label)) :
                label[i] = label[i]-1
        
        feature_tensor = torch.from_numpy(feature).float().clone()
        label_tensor = torch.tensor(label)
        
        # 명시적으로 대용량 딕셔너리 참조 해제 유도
        del fea, feature, label
        
        return feature_tensor, label_tensor
        

def main() :
    test_dataset = CustomDataset(path_label="/home/jjw/proj/spk_seg/dump/mfcc/val/label_format",
                                 path_fea="/home/jjw/proj/spk_seg/dump/mfcc/val/feature",
                                 input_size=20)
    print(next(iter(DataLoader(dataset=test_dataset,
                     batch_size=16)))[0])

if __name__ == "__main__" :
    main()