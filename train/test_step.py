import numpy as np
import os
from tqdm import tqdm
import pickle
import torch
import webrtcvad
import contextlib
import wave
import soundfile as sf

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def max_iter(list_a) :
    values, counts = np.unique(list_a, return_counts=True)
    index = np.argmax(counts)
    return values[index]


def frame_vad(wav_file, data_conf) :
    wav, sr = sf.read(os.path.join(data_conf['audio_path'],data_conf['test_name'],wav_file.split(".")[0]+'.wav'))
    wav_speech = get_speech_timestamps(wav, vad_model, sampling_rate=data_conf['sr'])
    wav_point = [wav_speech[0]['start'], wav_speech[-1]['end']]
    wav_vad = np.ones(len(wav))
    for i in range(len(wav)) :
        if i < wav_point[0] or i > wav_point[1] :
            wav_vad[i] = 0
    
    offset = 0
    vad_label=[]
    win_len = int(data_conf['sr']*data_conf['win_ms']/1000)
    hop_len = int(data_conf['sr']*data_conf['hop_ms']/1000)
    
    while offset+win_len <= len(wav):
        values, counts = np.unique(wav_vad[offset:offset+win_len], return_counts=True)
        index = np.argmax(counts)
        vad_label.append(values[index])
        offset += hop_len
    
    return vad_label

def check_vad_pred(pred_frame, vad_frame) :
    for i in range(len(vad_frame)) :
        pred_frame[i] = int(pred_frame[i]*vad_frame[i])
    return pred_frame

def test_model(model, path_test, path_result, train_conf, data_conf, device) :
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
            
        vad_label = frame_vad(wav_file, data_conf)
        pred_label = check_vad_pred(pred_label, vad_label)
        
        for i in range(len(pred_label)) :
            if train_conf['zero_class'] :
                cnt_label[wav_info['label'][i],pred_label[i]] += 1
            else :
                cnt_label[wav_info['label'][i]-1,pred_label[i]] += 1
    
    return frame_cnt, cnt_label

def print_result(cnt_list, title, train_conf) :
    print(cnt_list)
    print(f"\t{title}")
    print("\t\tACC", (cnt_list[0,0]+ cnt_list[1,1]+ cnt_list[2,2])/np.sum(cnt_list))
    
    weighted_f1 = 0
    for i in range(len(cnt_list)) :
        precision = cnt_list[i,i] / np.sum(cnt_list[:,i])
        recall = cnt_list[i,i] / np.sum(cnt_list[i,:])
        print(f"\n\t\tclass{i}, precision : {precision}, recall : {recall}")
        print(f"\t\t\tF1_score : {2*(precision*recall)/(precision+recall)}")
        weighted_f1 = 2*(precision*recall)/(precision+recall) * np.sum(cnt_list[i:]) / np.sum(cnt_list)
    print(f"\t\tweighted_f1 : {weighted_f1}")

def set_test(model, path_test, path_result, train_conf, data_conf, device) :
    model.eval()
    
    frame_cnt, cnt_label = test_model(model, path_test, path_result, train_conf, data_conf, device)
    
    #print_result(frame_cnt,"all_pred_frame_acc",train_conf)
    print_result(cnt_label,"wav_frame_acc",train_conf)

def main() :
    set_test
    
if __name__ == "__main__" :
    main()