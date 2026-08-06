import pickle
import os
from random import choice


def make_list(data_conf, train_conf) :
    
    # train data
    file_train_list = os.listdir(os.path.join(data_conf['data_path'], data_conf['train_name']))
    #cnt_file = int(len(file_train_list)*data_conf['use_rate'])
    cnt_file = int(data_conf['use_rate'])
    print(cnt_file)
    train_real_list = []
    fea_list = []
    for i in range(cnt_file) :
        random_wav = choice(file_train_list)
        while random_wav in train_real_list :
            random_wav = choice(file_train_list)
        random_wav = random_wav.split(".")[0]
        train_real_list.append(random_wav)
        
        
        with open(os.path.join(data_conf['data_path'],data_conf['train_name'],random_wav+".pickle"),'rb') as fr :
            fea = pickle.load(fr)
        
        for i in range(fea['mfcc'].shape[-1]) :
            i_num = []
            n_size = 0
            while len(i_num) < train_conf['input_size'] :
                if i+n_size >= fea['mfcc'].shape[-1] :
                    break
                if not train_conf['zero_class'] and fea['label'][i+n_size] == 0 :
                    n_size+=1
                    continue
                i_num.append(i+n_size)
                n_size+=1
                
            if len(i_num) == train_conf['input_size'] :
                fea_list.append({'file':random_wav,'num':i_num})
        
    os.system("mkdir -p %s" %(data_conf['dump_path']))
    os.system("rm %s" %(os.path.join(data_conf['dump_path'], 'train'+str(data_conf['use_rate'])+'.pickle')))
    
    with open(os.path.join(data_conf['dump_path'],'train'+str(data_conf['use_rate'])+'.pickle'),'wb') as fw :
        pickle.dump({'wav_list':train_real_list,'fea_list':fea_list},fw)
    print(data_conf['dump_path'], 'train'+str(data_conf['use_rate'])+'.pickle')
    
    
    
    
    
    # validation data
    file_val_list = os.listdir(os.path.join(data_conf['data_path'], data_conf['val_name']))
    cnt_file = int(len(file_val_list))
    val_real_list = []
    fea_list = []
    for i in range(cnt_file) :
        random_wav = choice(file_val_list)
        while random_wav in val_real_list :
            random_wav = choice(file_val_list)
        random_wav = random_wav.split(".")[0]
        val_real_list.append(random_wav)
        
        
        with open(os.path.join(data_conf['data_path'],data_conf['val_name'],random_wav+".pickle"),'rb') as fr :
            fea = pickle.load(fr)
        
        for i in range(fea['mfcc'].shape[-1]) :
            i_num = []
            n_size = 0
            while len(i_num) < train_conf['input_size'] :
                if i+n_size >= fea['mfcc'].shape[-1] :
                    break
                if not train_conf['zero_class'] and fea['label'][i+n_size] == 0 :
                    n_size+=1
                    continue
                i_num.append(i+n_size)
                n_size+=1
                
            if len(i_num) == train_conf['input_size'] :
                fea_list.append({'file':random_wav,'num':i_num})
        
    os.system("mkdir -p %s" %(data_conf['dump_path']))
    os.system("rm %s" %(os.path.join(data_conf['dump_path'], 'val'+str(data_conf['use_rate'])+'.pickle')))
    
    with open(os.path.join(data_conf['dump_path'], 'val'+str(data_conf['use_rate'])+'.pickle'),'wb') as fw :
        pickle.dump({'wav_list':val_real_list,'fea_list':fea_list},fw)
    
        

if __name__ == "__main__" :
    data_conf = {}
    make_list(data_conf)