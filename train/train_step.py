import torch.nn as nn
import torch
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm
import gc

def train(device, model, train_loader, val_loader, folder_save,epochs,lr, early_stop, num_early) :
    torch.cuda.init()
    torch.cuda.reset_max_memory_cached()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    early_list = []
    check_early = False
    best_loss = 100
    best_acc = 0
    gc.collect()
    for epoch in range(epochs) :
        torch.cuda.init()
        torch.cuda.empty_cache()
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Train") as pbar:
            for i, (inputs, targets) in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # [해결] 배치 종료 시 모든 관련 텐서/변수 참조 해제
                del loss, outputs, inputs, targets
        
        
        # validation
        gc.collect()
        model.eval()
        total_cnt = 0
        correct_cnt = 0
        loss_list = []
        with torch.no_grad() :
            for i_num, (inputs, targets) in tqdm(enumerate(val_loader)):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_list.append(loss)

                _, predicted = torch.max(outputs.data, 1)

                total_cnt += targets.size(0) * targets.size(1)
                correct_cnt += (predicted == targets).sum()

        loss_list = torch.tensor(loss_list)
        if torch.mean(loss_list) < best_loss :
            torch.save(model.state_dict(), f"{folder_save}/best_loss")
            best_loss = torch.mean(loss_list)
        if 100 * correct_cnt / total_cnt > best_acc :
            torch.save(model.state_dict(), f"{folder_save}/best_acc")
            best_acc = 100 * correct_cnt / total_cnt
            print(f"new best_acc {epoch}")
        print(f'Epoch {epoch}. loss : {torch.mean(loss_list)}, Model Accuracy: {100 * correct_cnt / total_cnt}%')
        
        gc.collect()
        # check train_acc
        if epoch%10 == 0 : 
            total_cnt = 0
            correct_cnt = 0
            with torch.no_grad() :
                for i, (inputs, targets) in tqdm(enumerate(train_loader)) :
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)

                    _, predicted = torch.max(outputs.data, 1)

                    total_cnt += targets.size(0) * targets.size(1)
                    correct_cnt += (predicted == targets).sum()
            print(f'Epoch {epoch}. train Model Accuracy: {100 * correct_cnt / total_cnt}%')
            gc.collect()
        if early_stop :
            count = 0
            early_list.append(torch.round(torch.mean(loss_list).cpu(),decimals=6))
            if len(early_list) > num_early :
                for i in range(1, num_early+1) :
                    if early_list[0] < early_list[i] :
                        count+=1
                if count >= num_early :
                    check_early = True
                    print(early_list)
                early_list = early_list[-1*num_early:]
                
        if check_early : break
        
        gc.collect()
        model.train()
    
    return model

def main() :
    pass
if __name__ == "__main__" :
    main()