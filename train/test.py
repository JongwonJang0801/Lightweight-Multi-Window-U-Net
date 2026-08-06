import numpy as np
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle

def draw_wav(wav1, wav2, filename='test.png') :
    #plt.plot(wav)
    plt.figure(figsize=(20,10))
    plt.subplot(211)
    plt.yticks([-1,0,1,2])
    plt.scatter(range(len(wav1)),wav1)
    
    
    plt.subplot(212)
    plt.yticks([-1,0,1,2])
    plt.scatter(range(len(wav2)),wav2)
    
    plt.savefig(filename)
    plt.close()

# input 1D (frame or wav)
def count_result(truth, pred, n_label) :
    cnt = np.zeros((n_label,n_label), dtype = int)
    for i in range(len(truth)) :
        cnt[truth[i],pred[i]] +=1
    return cnt

def count_result_wovad(truth, pred, n_label, vad) :
    cnt = np.zeros((n_label,n_label), dtype = int)
    for i in range(len(truth)) :
        if vad[i] == 0 :
            cnt[truth[i],pred[i]] +=1
    return cnt

def count_result_wovad_label(truth, pred, n_label) :
    cnt = np.zeros((n_label,n_label), dtype = int)
    for i in range(len(truth)) :
        if truth[i] == 0 : continue
        cnt[truth[i],pred[i]] +=1
    return cnt

def smooth_pred(pred,window=4) :
    temp = np.concatenate((np.zeros(window, dtype=int),
                          pred,
                          np.zeros(window, dtype=int)),dtype=int)
    for i in range(len(pred)) :
        pred[i] = round(np.mean(temp[i:i+window+window+1]),0)
    return pred

def print_metric(cnt_list) :
    print(cnt_list)
    print("acc :", (cnt_list[0,0]+cnt_list[1,1]+cnt_list[2,2])/np.sum(cnt_list)*100)
    print("VAD : MD : ", (cnt_list[1,0]+cnt_list[2,0])/np.sum(cnt_list[1:,:])*100)
    print("one_spk : MD : ", (cnt_list[1,0]+cnt_list[1,2])/np.sum(cnt_list[1,:])*100)
    print("mix_spk : MD : ", (cnt_list[2,0]+cnt_list[2,1])/np.sum(cnt_list[2:,:])*100)
    print()


def find_max_iter(list_a, n_class) :
    cnt = np.zeros(n_class)
    for i in list_a :
        cnt[i] = cnt[i] + 1
    
    max_cnt = 0
    for i in range(n_class) :
        if max_cnt <= cnt[i] :
            max_index = i
            max_cnt = cnt[i]
            
    return max_index

def test_model(dump_dir, fea_name, input_size, dir_result, model,device, model_conf) :
    
    # save result in matrix(true,pred)
    frame_cnt = np.zeros((model_conf['n_class'],model_conf['n_class']),dtype=int)
    cnt_wovad = np.zeros((model_conf['n_class'],model_conf['n_class']),dtype=int)
    cnt_wovad_label = np.zeros((model_conf['n_class'],model_conf['n_class']),dtype=int)
    
    os.system('mkdir -p %s'%(os.path.join(dir_result,'test_result')))
    for filename in tqdm(os.listdir(os.path.join(dump_dir,'info','test'))) :
        
        # load test info and fea
        with open(os.path.join(dump_dir,'info','test',filename), 'rb') as f :
            info = pickle.load(f)
        with open(os.path.join(dump_dir,fea_name,'test',filename), 'rb') as f :
            fea = pickle.load(f)
        
        wav_input = []
        wav_output = []
        
        for i in range(len(info['label'])-model_conf['input_size']+1) :
            
            # if zero_class / test without vad -1
            if model_conf['zero_class'] :
                '''
                if 
                num = 0
                input_one_sample = []
                while num < 40 and i+num < len(info['label']) :
                '''
                pass
            
            # make input and output(label)
            else :
                wav_input.append(fea[:, i:i+input_size])
                wav_output.append(info['label'][i:i+input_size])
        
        # make input and output to numpy
        wav_input = np.array(wav_input)[:, np.newaxis,:,:]
        wav_output = np.array(wav_output, dtype=int)
        
        # to use gpu, split data from batch_size
        iter_wav = (len(wav_input) // model_conf['batch_size']) + 1
        if len(wav_input) % model_conf['batch_size'] == 0 :
            iter_wav = 0
        test_pred = [[]for i in range(len(info['label']))]
        for n_batch in range(iter_wav) :
            model.to(device)
            #print(n_batch)
            #print(np.array(wav_input[n_batch*model_conf['batch_size']:(n_batch+1)*model_conf['batch_size']]).shape)
            pred_batch = model(torch.from_numpy(wav_input[n_batch*model_conf['batch_size']:(n_batch+1)*model_conf['batch_size']]).float().to(device))
            _, predicted = torch.max(pred_batch.data, 1)
            
            pred = np.array(predicted.cpu())
            for i in range(len(pred)) :
                for j in range(pred.shape[-1]) :
                    test_pred[(n_batch*model_conf['batch_size'])+i+j].append(pred[i,j])
        
        for i in range(len(test_pred)) :
            test_pred[i] = find_max_iter(test_pred[i], model_conf['n_class'])
        
        cnt = count_result(info['label'], np.array(test_pred), model_conf['n_class'])
        frame_cnt += cnt
        cnt = count_result_wovad(info['label'], np.array(test_pred), model_conf['n_class'], info['vad'])
        cnt_wovad += cnt
        cnt = count_result_wovad_label(info['label'], np.array(test_pred), model_conf['n_class'])
        cnt_wovad_label += cnt
        
        dirname = filename.split('.')[0]
        draw_wav(info['label'], np.array(test_pred),
                 filename=os.path.join(dir_result,'test_result',f'{dirname}.png'))
    
    return frame_cnt, cnt_wovad, cnt_wovad_label

def set_test(model, dir_result, data_conf, model_conf, device) :
    # load model
    model.eval()
    
    # test model and return result
    frame_cnt, cnt_wovad, cnt_wovad_label = test_model(data_conf['dir_dump'],
                                               data_conf['fea_name'],
                                               model_conf['input_size'],
                                               dir_result, model, device, model_conf)
    
    # print metric
    print("ACC of result")
    print_metric(frame_cnt)
    print("ACC with out vad with mixvad")
    print_metric(cnt_wovad)
    print("ACC with out vad with labelvad")
    print_metric(cnt_wovad_label)
    
    
def main() :
    set_test()
if __name__ == "__main__" :
    main()