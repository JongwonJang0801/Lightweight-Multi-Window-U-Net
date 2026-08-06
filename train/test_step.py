import numpy as np
import os
from tqdm import tqdm
import pickle
import torch

def max_iter(list_a) :
    values, counts = np.unique(list_a, return_counts=True)
    index = np.argmax(counts)
    return values[index]

def test_model(model, path_test, path_result, train_conf, device) :
    frame_cnt = np.zeros((train_conf['n_class'],train_conf['n_class']),dtype=int)
    cnt_label = np.zeros((train_conf['n_class'],train_conf['n_class']),dtype=int)
    
    os.system('mkdir -p %s' %(path_result))
    for wav_file in tqdm(os.listdir(path_test)) :
        with open(os.path.join(path_test,wav_file),'rb') as f :
            wav_info = pickle.load(f)
        x = []
        y = []
        for i in range(len(wav_info['label'])-train_conf['input_size']+1) :
            x.append(wav_info['mfcc'][:,i:i+train_conf['input_size']])
            y.append(wav_info['label'][i:i+train_conf['input_size']])
        
        x = np.array(x,dtype=np.float64)[:,np.newaxis,...]
        y = np.array(y,dtype=int)
        pred_label = [[] for i in range(wav_info['mfcc'].shape[-1])]
        
        for n_batch in range(int(len(x)/train_conf['batch_size'])+1) :
            if len(x) == n_batch*train_conf['batch_size'] : break
            model.to(device)
            pred_batch = model(torch.from_numpy(x[n_batch*train_conf['batch_size']:(n_batch+1)*train_conf['batch_size']]).float().to(device))
            _, predicted = torch.max(pred_batch.data,1)
            
            pred = np.array(predicted.cpu())
            for i in range(len(pred)) :
                for j in range(pred.shape[-1]) :
                    pred_label[(n_batch*train_conf['batch_size'])+i+j].append(pred[i,j])
                    if train_conf['zero_class'] :
                        frame_cnt[y[(n_batch*train_conf['batch_size'])+i,j],pred[i,j]] +=1
                    else :
                        frame_cnt[y[(n_batch*train_conf['batch_size'])+i,j]-1,pred[i,j]] +=1
                    
        for i in range(len(pred_label)) :
            pred_label[i] = max_iter(pred_label[i])
        
        for i in range(len(pred_label)) :
            if train_conf['zero_class'] :
                cnt_label[y[i],pred_label[i]]
            else :
                cnt_label[wav_info['label'][i]-1,pred_label[i]] += 1
    
    return frame_cnt, cnt_label

def print_result(cnt_list, title, train_conf) :
    print(cnt_list)
    print(f"\t{title}")
    if train_conf['zero_class'] :
        print("\t\tACC", (cnt_list[0,0]+ cnt_list[1,1]+ cnt_list[2,2])/np.sum(cnt_list))
        print("\t\tACC_vad", (cnt_list[1,1]+ cnt_list[2,2])/np.sum(cnt_list[1:,:]))
    else :
        print("\t\tACC", (cnt_list[0,0]+ cnt_list[1,1]+ cnt_list[2,2])/np.sum(cnt_list[:-1,:]))

def set_test(model, path_test, path_result, train_conf, device) :
    model.eval()
    
    frame_cnt, cnt_label = test_model(model, path_test, path_result, train_conf, device)
    
    print_result(frame_cnt,"all_pred_frame_acc",train_conf)
    print_result(cnt_label,"wav_frame_acc",train_conf)

def main() :
    set_test
    
if __name__ == "__main__" :
    main()