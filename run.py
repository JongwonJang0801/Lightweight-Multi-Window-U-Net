import os
import yaml
import time
import torch
from torch.utils.data import DataLoader
from torchsummary import summary as summary
import importlib

from train import make_train_list, dataset, train_step, test_step

# 그래프
def load_module_func(module_name):
    mod = importlib.import_module(module_name)
    return mod

# setting step , load config file
start_step = 1
pass_step = [0,1]
'''
Step 1:load_data
Step 2: ready setting for train.
Step 3: Start train.
Step 4: Test model.
'''

with open('local/conf/conf.yaml') as f:
    conf = yaml.full_load(f)
f.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mymodel = load_module_func('model.%s'%(conf['model_conf']['model_name']))

def print_log(s) :
    print(time.strftime('%x %X'),"||",s)

def load_dataloader(dataset, batch_size) :
    dataloader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    drop_last=True,
                                    pin_memory=True,
                                    )
    return dataloader

def main() :
    if conf['fea_conf']['fea_name'] == 'mfcc' :
        n_fea = conf['fea_conf']['n_fea']*3
    else :
        n_fea = conf['fea_conf']['n_fea']
        
        
    if 1 >= start_step and not 1 in pass_step :
        print("=================================")
        print_log("Step 1:load_data")
        
        make_train_list.make_list(conf['data_conf'], conf['train_conf'])
    
    
    if 2 >= start_step and not 2 in pass_step :
        print("=================================")
        print_log("Step 2: ready setting for train.")
        print_log("\tload train_dataset")
        train_dataset = dataset.CustomDataset(path_dump=os.path.join(conf['data_conf']['dump_path'],'train'+str(conf['data_conf']['use_rate'])+'.pickle'),
                                              path_fea=os.path.join(conf['data_conf']['data_path'],conf['data_conf']['train_name']),
                                              input_size=conf['train_conf']['input_size'],
                                              zero_class=conf['train_conf']['zero_class'],
        )
        
        val_dataset = dataset.CustomDataset(path_dump=os.path.join(conf['data_conf']['dump_path'],'val'+str(conf['data_conf']['use_rate'])+'.pickle'),
                                              path_fea=os.path.join(conf['data_conf']['data_path'],conf['data_conf']['val_name']),
                                              input_size=conf['train_conf']['input_size'],
                                              zero_class=conf['train_conf']['zero_class'],
        )

        train_loader = load_dataloader(train_dataset,conf['train_conf']['batch_size'])
        val_loader = load_dataloader(val_dataset,conf['train_conf']['batch_size'])
        
        print("\tfinish load data")
        model=mymodel.myModel(conf['train_conf']['input_size'],
                              n_fea,
                              conf['train_conf']['n_class'],
                              conf['encoder_conf'])
        #print(model)
        model.to(device)
        summary(model, (1,48,64)) # (model, input_size)
        print(next(iter(train_loader))[0].size())


    if 3 >= start_step and not 3 in pass_step :
        print("=================================")
        print_log("Step 3: Start train.")
        
        ###############################################################################
        folder_save = os.path.join(conf['data_conf']['dump_path'],time.strftime('%m%d')+'_'+conf['model_conf']['model_name'])
        os.system(f"mkdir -p %s"%(folder_save))
        model = train_step.train(device, model, train_loader, val_loader, folder_save,
                                 conf['train_conf']['epochs'],
                                 conf['train_conf']['lr'],
                                 conf['train_conf']['early_stop'],
                                 conf['train_conf']['num_early'],
        )
        
        torch.save(model.state_dict(), f"{folder_save}/final")
        model.to("cpu")
        with open(os.path.join(folder_save,'conf.yaml'), 'w') as f:
            yaml.dump(conf, f)
    
    
    if 4 >= start_step and not 4 in pass_step :
        print("=================================")
        print_log("Step 4: Test model.")
        if conf['train_conf']['only_test'] :
            model=mymodel.myModel(conf['train_conf']['input_size'],
                                n_fea,
                                conf['train_conf']['n_class'],
                                conf['encoder_conf'])
            
            model.load_state_dict(torch.load(os.path.join(conf['data_conf']['dump_path'],
                                                          conf['train_conf']['test_folder'],
                                                          'final')))
            folder_save = os.path.join(conf['data_conf']['dump_path'], conf['train_conf']['test_folder'])
        
        
        model.load_state_dict(torch.load(os.path.join(folder_save,
                                                        'best_acc')))
        test_step.set_test(model,
                           path_test=os.path.join(conf['data_conf']['data_path'],conf['data_conf']['test_name']),
                           path_result= folder_save,
                           train_conf = conf['train_conf'],
                           device = device,)


if __name__ == "__main__" :
    main()