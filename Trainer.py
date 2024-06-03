import ModelModuleBlue
import ModelModuleLLama
import torch
import os
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.nn as nn
#import deepspeed
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import LinearLR
from tensorboardX import SummaryWriter
from time import time
from calflops import calculate_flops
import re

def remove_starting_symbols(input_string):
    pattern = re.compile(r'^[^a-zA-Z\u4e00-\u9fa5]+')
    cleaned_string = pattern.sub('', input_string)
    return cleaned_string


class Trainer:
    def __init__(self, model:[ ModelModuleBlue.ModelModule, ModelModuleLLama.ModelModule] , DataLoaders: list, tokenizer: AutoTokenizer, hParams: dict, config: dict) -> None:
        self.__config = config
        if config['synced']:
            self.__localRank = int(os.environ['LOCAL_RANK'])
        
        if config['action'] == 'train':
            self.__logger = SummaryWriter(config['logSavedPath'])
            
        #data loader
        self.__trainDataLoader = DataLoaders[0]
        self.__valDataLoader = DataLoaders[1]
        self.__testDataLoader = DataLoaders[2]
        
        #(h)params
        self.__modelSavedPath = config['modelSavedPath']
        self.__evalParms = hParams['evalParams']
        self.__tokenizer = tokenizer
        self.__epochs = hParams['epochs']
        self.__lr = hParams['lr']
        self.__device = torch.device("cuda:{0}".format(self.__localRank)) if config['synced'] else 'cuda'
        
        #model
        self.__model = model
        
        #deep speed
        self.__dsConfig = {"train_batch_size": 1,
                     "optimizer": {
                         "type": "Adam",
                         "params": {
                             "lr": self.__lr,
                        }
                    }
                }
        
        #AMP settings
        self.__scaler = amp.GradScaler()
        
        # log
        self.__bestModel = 100
        self.__evalSteps = 0
    
    def __Log(self, tag, scalar, batchIdx) -> None:
        self.__logger.add_scalar(tag=tag, scalar_value=scalar, global_step=batchIdx)
    
    def __DistributedGather(self, metric: torch.Tensor) -> torch.Tensor:
        gatheredTensors = [metric.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(gatheredTensors, metric)
        return torch.cat(gatheredTensors, dim=0)
    
    def Train(self) -> None:
        if self.__config['synced']:
            self.__model.FreezeBackbone()
            self.__model.SeperateEmbedding()
            if self.__config['resume']:
                self.__model.SetTrainedEmbedding()
            self.__model = self.__model.to(self.__device)
            self.__model = DDP(self.__model,
                            device_ids = [self.__localRank],
                            broadcast_buffers = False,
                            find_unused_parameters = False)
            self.__model = nn.SyncBatchNorm.convert_sync_batchnorm(self.__model).to(self.__device)
            _optimizer = ZeroRedundancyOptimizer(filter(lambda p: p.requires_grad==True ,self.__model.parameters()),
                                                optimizer_class = Adam,
                                                lr = self.__lr)
        else:
            self.__model.FreezeBackbone()
            self.__model.SeperateEmbedding()
            if self.__config['resume']:
                self.__model.SetTrainedEmbedding()
            self.__model = self.__model.to(self.__device)
            _optimizer = Adam(filter(lambda p: p.requires_grad==True ,self.__model.parameters()), lr=self.__lr)
        
        # self.__model, _optimizer, _, _ = deepspeed.initialize(model = self.__model,
        #                                                       optimizer = _optimizer, 
        #                                                       model_parameters = self.__model.parameters(),
        #                                                       config_params = self.__dsConfig)
    
        _lrScheduler = LinearLR(_optimizer, total_iters= self.__epochs)
        idx = 0
        #self.Evalueate()
        for epoch in range(self.__epochs):
            self.__model.train()
            for batch in tqdm(self.__trainDataLoader, desc='train'):
                if batch['op'] == ['recursive'] or batch['op'] == ['concat']:
                    for i in range(len(batch['inputs']['input_ids'])):
                        batch['inputs']['input_ids'][i] = batch['inputs']['input_ids'][i].to(self.__device)
                    batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
       
                else:
                    batch['inputs'] = {k: v.to(self.__device) for k, v in batch['inputs'].items() if isinstance(v, torch.Tensor)}
                    batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
                # with amp.autocast():
                loss = self.__model(**batch)
                if self.__config['synced'] and self.__localRank == 0:
                    self.__Log('train/loss/{}'.format(batch['op']), loss.item(), idx)
                elif not self.__config['synced']:
                    self.__Log('train/loss/{}'.format(batch['op']), loss.item(), idx)
                idx += 1
                
                if self.__config['synced']:
                    # self.__scaler.scale(loss).backward()
                    # self.__scaler.step(_optimizer)
                    # self.__scaler.update()
                    loss.backward()
                    _optimizer.step()
                    _optimizer.zero_grad(set_to_none=True)
                    if idx and idx % self.__evalParms['evalStep'] == 0:
                        self.Evalueate()
                        self.__model.train()
                else:
                    loss /= 4
                    self.__scaler.scale(loss).backward()
                    if idx % 4 == 0 or idx + 4 == len(self.__trainDataLoader):
                        _optimizer.step()
                        # self.__scaler.step(_optimizer)
                        # self.__scaler.update()
                        _optimizer.zero_grad(set_to_none=True)
                    if idx and idx % self.__evalParms['evalStep'] == 0:
                        self.Evalueate()
                        self.__model.train()    
                #end if synced
            #end for batch
            _lrScheduler.step()
        #end for epoch
        self.Evalueate()
    
    def ComputeTime(self) -> None:
        self.__model.FreezeBackbone()
        self.__model.SeperateEmbedding()
        self.__model.SetTrainedEmbedding()
        self.__model.to(self.__device)
        self.__model.eval()

        with torch.no_grad():
            TS = time()
            flops, macs, params = calculate_flops(model=self.__model,
                                    print_results=True,
                                    forward_mode='timecompute',
                                    print_detailed=False)
            ES = time()
            print(ES-TS, flops, macs, params)
                    
    def Evalueate(self):
        self.__model.to(self.__device)
        self.__model.eval()

        lossCollector = []
        with torch.no_grad():
            for batch in tqdm(self.__valDataLoader, desc='eval'):
                if batch['op'] == ['recursive'] or batch['op'] == ['concat']:
                    for i in range(len(batch['inputs']['input_ids'])):
                        batch['inputs']['input_ids'][i] = batch['inputs']['input_ids'][i].to(self.__device)
                    batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
                else:
                    batch['inputs'] = {k: v.to(self.__device) for k, v in batch['inputs'].items() if isinstance(v, torch.Tensor)}
                    batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
                loss = self.__model(**batch)
                lossCollector.append(loss.item())
        if self.__config['synced']:
            globalLoss = self.__DistributedGather(torch.tensor(lossCollector, device=self.__localRank))
        else:
            globalLoss = torch.tensor(lossCollector, device=self.__device)
            
        if not self.__config['synced'] or self.__config['synced'] and self.__localRank == 0:
            avgLoss =globalLoss.mean().item()
            print('eval step {0}:average loss:{1}'.format(self.__evalSteps, avgLoss))
            self.__Log('eval/loss', avgLoss, self.__evalSteps)
            if avgLoss < self.__bestModel:
                self.__bestModel = avgLoss
                if self.__config['synced']:
                    self.__model.module.SaveTrainedModel(avgLoss, self.__modelSavedPath + 'Step{0}'.format(self.__evalSteps))
                else:
                    self.__model.SaveTrainedModel(avgLoss, self.__modelSavedPath + '/Step{0}'.format(self.__evalSteps))  
        self.__evalSteps += 1
             
    def Test(self):
        self.__model.FreezeBackbone()
        self.__model.SeperateEmbedding()
        self.__model.SetTrainedEmbedding()
        self.__model.to(self.__device)
        self.__model.eval()
        partialPreds = []
        dt, at = 0, 0
        with torch.no_grad():
            for ratio in [4, 8, 12, 16, 1024]:
                self.__testDataLoader.dataset.SetRatio(ratio)
                for batch in tqdm(self.__testDataLoader, desc='inference'):
                    batch['inputs'] = {k: v.to(self.__device) for k, v in batch['inputs'].items() if isinstance(v, torch.Tensor)}
                    batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
                    preds, bdt, bat = self.__model.inference(**batch)
                    dt += bdt
                    at += bat
                    partialPreds.extend(preds.split(1))
                if self.__config['synced']:
                    print(f'RANK: {self.__localRank}||All time cost: {at}, Decoding Time Cost: {dt}')
                    
                partialPreds = torch.cat(partialPreds).to(self.__device)
                if self.__config['synced']:
                    globalPreds = self.__DistributedGather(partialPreds)
                    partialPreds = []
                else:
                    globalPreds = partialPreds
                if not self.__config['synced'] or self.__localRank == 0:
                    seqs = self.__tokenizer.batch_decode(globalPreds, skip_special_tokens=True)
                    seqs = [seq.encode(errors='ignore').decode('utf-8') for seq in seqs]
                    seqs = [remove_starting_symbols(seq) for seq in seqs]
                    print(len(seqs))
                    fh = open(f'/data1/gj/PromptCompression/ratios/blue/{ratio}.txt', 'w', encoding = 'utf-8')
                    for seq in seqs:
                        fh.write(seq.replace('\n', '').replace('\r', '') + '\n')
                    fh.close()
                  
    def BaselineCOT(self):
        self.__model.to(self.__device)
        self.__model.eval()
        partialProbs = []

        with torch.no_grad():
            for batch in tqdm(self.__testDataLoader, desc='baseline'):
                batch['inputs']['input_ids'] = batch['inputs']['input_ids'].to(self.__device)
                probs = self.__model.naivecot(**batch)
                partialProbs.append(probs)
        
        partialProbs = torch.cat(partialProbs).to(self.__device)
        if self.__config['synced']:
            globalProbs = self.__DistributedGather(partialProbs)
        else:
            globalProbs = partialProbs
        if not self.__config['synced'] or self.__localRank == 0:
            indices = torch.max(globalProbs, dim=-1)[1]
            indices = [chr(ord('A') + i) for i in indices]
            fh = open('/data1/gj/PromptCompression-vicuna/cot/check-baseline-retrieve_2.txt', 'w')
            fh.write('\n'.join(indices))
            fh.close()
    
    def ContextCOT(self):
        self.__model.FreezeBackbone()
        self.__model.SeperateEmbedding()
        self.__model.SetTrainedEmbedding()
        self.__model.to(self.__device)
        self.__model.eval()
        partialProbs = []
        with torch.no_grad():
            for batch in tqdm(self.__testDataLoader, desc='cot'):
                for i in range(len(batch['inputs']['input_ids'])):
                    batch['inputs']['input_ids'][i] = batch['inputs']['input_ids'][i].to(self.__device)
                batch['c_inputs']['input_ids'] = batch['c_inputs']['input_ids'].to(self.__device)
                probs = self.__model.cprsedcot(**batch)
                partialProbs.append(probs)
        
        partialProbs = torch.cat(partialProbs).to(self.__device)
        if self.__config['synced']:
            globalProbs = self.__DistributedGather(partialProbs)
        else:
            globalProbs = partialProbs
        if not self.__config['synced'] or self.__localRank == 0:
            indices = torch.max(globalProbs, dim=-1)[1]
            indices = [chr(ord('A') + i) for i in indices]
            fh = open('/data1/gj/PromptCompression-vicuna/cot/cprsed-retrieve_two-add.txt', 'w')
            fh.write('\n'.join(indices))
            fh.close()
                   
    def Baseline(self):
        self.__model.to(self.__device)
        self.__model.eval()
        partialPreds = []
        at = 0
        with torch.no_grad():
            for batch in tqdm(self.__testDataLoader, desc='baseline'):
                if isinstance(batch['inputs']['input_ids'], list):
                    for i in range(len(batch['inputs']['input_ids'])):
                        batch['inputs']['input_ids'][i] = batch['inputs']['input_ids'][i].to(self.__device)
                else:
                    batch['inputs']['input_ids'] = batch['inputs']['input_ids'].to(self.__device)
                preds, bat = self.__model.baseline(**batch)
                partialPreds.extend(preds.split(1))
                at += bat
        if self.__config['synced']:
            print(f'RANK: {self.__localRank}||time cost: {at}s')
        partialPreds = torch.cat(partialPreds).to(self.__device)
        if self.__config['synced']:
            globalPreds = self.__DistributedGather(partialPreds)
        else:
            globalPreds = partialPreds
        if not self.__config['synced'] or self.__localRank == 0:
            seqs = self.__tokenizer.batch_decode(globalPreds, skip_special_tokens=True)
            seqs = [seq.encode(errors='ignore').decode('utf-8') for seq in seqs]
            seqs = [remove_starting_symbols(seq) for seq in seqs]
            
            print(len(seqs))
            fh = open('/data1/gj/PromptCompression/promptcompressor-result/blue/duc.txt', 'w', encoding = 'utf-8')
            for seq in seqs:
                fh.write(seq.replace('\n', '').replace('\r', '') + '\n')
            fh.close()
            
    def Recursive(self):
        self.__model.FreezeBackbone()
        self.__model.SeperateEmbedding()
        self.__model.SetTrainedEmbedding()
        self.__model.to(self.__device)
        self.__model.eval()
        partialPreds = []
        with torch.no_grad():
            for batch in tqdm(self.__testDataLoader, desc='recursive'):
                
                for i in range(len(batch['inputs']['input_ids'])):
                    batch['inputs']['input_ids'][i] = batch['inputs']['input_ids'][i].to(self.__device)

                batch['c_inputs']['input_ids'] = batch['c_inputs']['input_ids'].to(self.__device)
                
                preds = self.__model.recursive(**batch)
                partialPreds.append(preds.view(1, -1))

        partialPreds = torch.cat(partialPreds).to(self.__device)
        if self.__config['synced']:
            globalPreds = self.__DistributedGather(partialPreds)
        else:
            globalPreds = partialPreds
        if not self.__config['synced'] or self.__localRank == 0:
            seqs = self.__tokenizer.batch_decode(globalPreds, skip_special_tokens=True)
            fh = open('/data1/gj/PromptCompression-vicuna/experiments/result/cprsed_result_mix_arxiv_greedy_float_rec_fix_32.txt', 'w', encoding = 'utf-8')
            for seq in seqs:
                fh.write(seq.replace('\n', '') + '\n')
            fh.close()
            
    def Context(self):
        self.__model.FreezeBackbone()
        self.__model.SeperateEmbedding()
        self.__model.SetTrainedEmbedding()
        self.__model.to(self.__device)
        self.__model.eval()
        partialPreds = []
        dt, at = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.__testDataLoader, desc='recursive'):
                for i in range(len(batch['inputs']['input_ids'])):
                    batch['inputs']['input_ids'][i] = batch['inputs']['input_ids'][i].to(self.__device)
                batch['c_inputs']['input_ids'] = batch['c_inputs']['input_ids'].to(self.__device)
                
                preds, bdt, bat  = self.__model.recursive(**batch)
                partialPreds.append(preds.view(1, -1))
                dt += bdt
                at += bat
        if self.__config['synced']:
            print(f'RANK: {self.__localRank}||All time cost: {at}, Decoding Time Cost: {dt}')
        
        partialPreds = torch.cat(partialPreds).to(self.__device)
        if self.__config['synced']:
            globalPreds = self.__DistributedGather(partialPreds)
        else:
            globalPreds = partialPreds
        if not self.__config['synced'] or self.__localRank == 0:
            seqs = self.__tokenizer.batch_decode(globalPreds, skip_special_tokens=True)
            fh = open('/data1/gj/PromptCompression/blue/context/context-xsum-rerank-one-shot-ratio=12-new11111', 'w', encoding = 'utf-8')
            for seq in seqs:
                fh.write(seq.replace('\n', '').replace('\r', '') + '\n')
            fh.close()
    
    def QFS(self):
        self.__model.FreezeBackbone()
        self.__model.SeperateEmbedding()
        self.__model.SetTrainedEmbedding()
        self.__model.to(self.__device)
        self.__model.eval()
        partialPreds = []
        dt, at = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.__testDataLoader, desc='qfs'): 
                if isinstance(batch['inputs']['input_ids'], list):
                    for i in range(len(batch['inputs']['input_ids'])):
                        batch['inputs']['input_ids'][i] = batch['inputs']['input_ids'][i].to(self.__device)  
                else:
                    batch['inputs']['input_ids'] = batch['inputs']['input_ids'].to(self.__device)

                batch['c_inputs']['input_ids'] = batch['c_inputs']['input_ids'].to(self.__device)
                
                if isinstance(batch['inputs']['input_ids'], list):
                    preds, bdt, bat = self.__model.recursive(**batch)
                else:
                    preds, bdt, bat = self.__model.inference(**batch)
                partialPreds.extend(preds.split(1))
                dt += bdt
                at += bat
        if self.__config['synced']:
            print(f'RANK: {self.__localRank}||All time cost: {at}, Decoding Time Cost: {dt}')
        partialPreds = torch.cat(partialPreds).to(self.__device)
        if self.__config['synced']:
            globalPreds = self.__DistributedGather(partialPreds)
        else:
            globalPreds = partialPreds
        if not self.__config['synced'] or self.__localRank == 0:
            seqs = self.__tokenizer.batch_decode(globalPreds, skip_special_tokens=True)
            seqs = [seq.encode(errors='ignore').decode('utf-8') for seq in seqs]
            seqs = [remove_starting_symbols(seq) for seq in seqs]
            
            print(len(seqs))
            fh = open('/data1/gj/PromptCompression/blue/qfs/qfs-ratio=12-partial', 'w', encoding = 'utf-8')
            for seq in seqs:
                fh.write(seq.replace('\n', '') + '\n')
            fh.close()
            