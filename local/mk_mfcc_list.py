import numpy as np
from multiprocessing import Process, Manager
import os
import time
from random import choice
import soundfile as sf
import pickle
import librosa

def wav2mfcc(wav, win_ms=25,hop_ms=10,sr=16000, n_mfcc=16) :
    win_len = int(sr*win_ms/1000)
    hop_len = int(sr*hop_ms/1000)
    mfcc = librosa.feature.mfcc(y=wav,
                                n_mfcc=n_mfcc,
                                hop_length=hop_len,
                                n_fft=win_len,
                                sr=sr,
                                center=False)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate((mfcc,mfcc_delta,mfcc_delta2))

def load_spk_list(p_list, all_list, data_path, save_path, win_ms=25,hop_ms=10,sr=16000, n_mfcc=16) :
    
    with open(os.path.join(data_path,'points.pickle'),'rb') as fr :
        points = pickle.load(fr)
    
    for wav_file in p_list :
        if not wav_file.endswith(".wav") : continue
        wav,sr = sf.read(os.path.join(data_path,wav_file))
        mfcc = wav2mfcc(wav)
        n_spk = np.zeros(len(wav))
        data_dic = points[wav_file.split(".")[0]]
        
        n_spk[data_dic['wav1'][0]:data_dic['wav1'][1]] += 1
        n_spk[data_dic['wav2'][0]:data_dic['wav2'][1]] += 1
        
        offset = 0
        label=[]
        win_len = int(sr*win_ms/1000)
        hop_len = int(sr*hop_ms/1000)
        
        while offset+win_len <= len(wav):
            values, counts = np.unique(n_spk[offset:offset+win_len], return_counts=True)
            index = np.argmax(counts)
            label.append(values[index])
            offset += hop_len
        
        label = np.array(label,dtype=int)
        save_dict = {'mfcc':mfcc, 'label':label}
        
        with open(os.path.join(save_path,wav_file.split(".")[0]+'.pickle'),'wb') as fw :
            pickle.dump(save_dict,fw)
            
        with open(os.path.join(save_path,wav_file.split(".")[0]+'.pickle'),'rb') as fr :
            test_dict = pickle.load(fr)
            

def main() :
    data_path = "../data/wsj_mix_0.3"
    folder_list=["si_dt_05","si_et_05","si_tr_s"]
    save_path = "../data/mfcc/wsj_mix_0.3"
    os.system(f"mkdir -p {save_path}")

    for folder in folder_list :
        os.system(f"mkdir -p {os.path.join(save_path,folder)}")
        print(f"{time.strftime('%x %X')} || Strat {folder} mix")
        
        processes = []
        num_cpu = 19
        
        # slicing spk to multiprocess
        wav_list = os.listdir(os.path.join(data_path,folder))
        p_num_wav = (len(wav_list)//num_cpu)+1
        
        # load each process
        for num in range(num_cpu) :
            p = Process(target=load_spk_list,
                        args=(wav_list[num*p_num_wav:(num+1)*p_num_wav],
                            wav_list,
                            os.path.join(data_path,folder),
                            os.path.join(save_path,folder),
                            )
                        )
            processes.append(p)
            p.start()
                
        for p in processes :
            p.join()
 

if __name__ == "__main__" :
    '''
    wav1 = "../../../dataset/LibriSpeech/train-clean-360/2774/131722/2774-131722-0013.flac"
    wav2 = "../../../dataset/LibriSpeech/train-clean-360/2397/162238/2397-162238-0044.flac"
    over_rate = [0.3,0.5]
    mix_libri(wav2,wav1, over_rate)
    '''
    main()