import numpy as np
from multiprocessing import Process, Manager
import os
import time
from random import choice
import soundfile as sf
import torch
import pickle

# 모델과 유틸리티 함수 로드 (인터넷 연결 필요, 캐시됨)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def vad_model(wav, sr) :
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
    return speech_timestamps

def mix_libri(wav1, wav2, over_rate) :
    s1,sr = sf.read(wav1)
    s2,sr = sf.read(wav2)
    s1_speech = vad_model(s1,sr)
    s2_speech = vad_model(s2,sr)
    s1_point = [s1_speech[0]['start'],s1_speech[-1]['end']]
    s2_point = [s2_speech[0]['start'],s2_speech[-1]['end']]
    
    mix_point = choice(range(int(len(s1)*over_rate[0]), int(len(s1)*over_rate[-1])))
    
    if len(s1) - mix_point > len(s2) :
        mix_wav = np.zeros(len(s1), np.float64)
    else :
        mix_wav = np.zeros(mix_point+len(s2), np.float64)
    
    mix_wav[0:len(s1)] += s1
    mix_wav[mix_point:mix_point+len(s2)] += s2
    s2_point = [s2_point[0]+mix_point,s2_point[-1]+mix_point]
    
    data = {'wav1':s1_point, 'wav2':s2_point, 'mix_point':mix_point}
    return mix_wav, data


def random_ex(full_list, n=3) :
    ex= []
    cnt = n
    if len(full_list) <= 3 :
        cnt = len(full_list)-1
    for i in range(cnt) :
        file = choice(full_list)
        while file in ex or not file.endswith("flac"):
            file = choice(full_list)
        ex.append(file)
    return ex

def load_spk_list(p_spk_list, spk_list, libri_path, over_rate, save_path, data_dict) :
    for spk1 in p_spk_list :
        
        random_wavs = os.listdir(os.path.join(libri_path,spk1))
        for wav1 in random_wavs :
            if not wav1.endswith(".wav") : continue
            
            # random seclect wav2
            spk2 = choice(spk_list)
            while spk1 == spk2 :
                spk2 = choice(spk_list)
            wav2 = choice(os.listdir(os.path.join(libri_path,spk2)))
            while not wav2.endswith('.wav') :
                wav2 = choice(os.listdir(os.path.join(libri_path,spk2)))
            
            mix_wav, points = mix_libri(os.path.join(libri_path,spk1,wav1),
                                        os.path.join(libri_path,spk2,wav2),
                                        over_rate)
            
            mix_file = wav1.split(".")[0]+'_'+wav2.split(".")[0]
            sf.write(os.path.join(save_path,mix_file+'.wav'),
                        mix_wav,
                        16000,
                        format='WAV'
                        )
            data_dict[mix_file] = points
            

def main() :
    data_path = "../../../dataset/wsj0_wav/wsj0"
    folder_list=["si_dt_05","si_et_05","si_tr_s"]
    over_rate = [0.3,0.5]
    save_path = "../data/wsj_mix_0.3"
    os.system(f"mkdir -p {save_path}")

    for folder in folder_list :
        os.system(f"mkdir -p {os.path.join(save_path,folder)}")
        
        with Manager() as manager :
            data_dict = manager.dict()
            
            #data_dict = {}
            print(f"{time.strftime('%x %X')} || Strat {folder} mix")
            processes = []
            num_cpu = 19
            
            # slicing spk to multiprocess
            spk_list = os.listdir(os.path.join(data_path,folder))
            p_num_spk = (len(spk_list)//num_cpu)+1
            
            # load each process
            for num in range(num_cpu) :
                p = Process(target=load_spk_list,
                            args=(spk_list[num*p_num_spk:(num+1)*p_num_spk],
                                spk_list,
                                os.path.join(data_path,folder),
                                over_rate,
                                os.path.join(save_path,folder),
                                data_dict
                                )
                            )
                processes.append(p)
                p.start()
                
            for p in processes :
                p.join()
            
            with open(os.path.join(save_path,folder,'points.pickle'),'wb') as fw :
                normal_dict = dict(data_dict)
                pickle.dump(normal_dict,fw)
            
        '''
        with open(os.path.join(save_path,folder,'points.pickle'),'rb') as fr :
            test_dict = pickle.load(fr)
        '''


if __name__ == "__main__" :
    '''
    wav1 = "../../../dataset/LibriSpeech/train-clean-360/2774/131722/2774-131722-0013.flac"
    wav2 = "../../../dataset/LibriSpeech/train-clean-360/2397/162238/2397-162238-0044.flac"
    over_rate = [0.3,0.5]
    mix_libri(wav2,wav1, over_rate)
    '''
    main()