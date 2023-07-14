import ModelModule
import torch
import os
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import LinearLR
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, model: ModelModule.ModelModule , DataLoaders: list, tokenizer: AutoTokenizer, hParams: dict, config: dict) -> None:
        self.__config = config
        if config['synced']:
            self.__localRank = int(os.environ['LOCAL_RANK'])
            
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
        self.__logger = SummaryWriter(config['logSavedPath'])
        self.__device = torch.device("cuda:{0}".format(self.__localRank)) if config['synced'] else 'cuda'
        
        #model
        self.__model = model
        
        #AMP settings
        self.__scaler = amp.GradScaler()
        
        # log
        self.__bestModel = 100
        self.__evalSteps = 0
        
    def __Decoder(self, preds: torch.Tensor) -> list:
        preds = self.__tokenizer.batch_decode(preds, 
                                              skip_special_tokens = True,
                                              clean_up_tokenization_spaces = True)
        return preds
    
    def __Log(self, tag, scalar, batchIdx) -> None:
        self.__logger.add_scalar(tag=tag, scalar_value=scalar, global_step=batchIdx)
    
    def __DistributedGather(self, metric: torch.Tensor) -> torch.Tensor:
        gatheredTensors = [metric.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(gatheredTensors, metric)
        return torch.cat(gatheredTensors, dim=0)
    
    def Train(self) -> None:
        if self.__config['synced']:
            self.__model.FreezeBackbone()
            self.__model = DDP(self.__model.to(self.__device),
                            device_ids = [self.__localRank],
                            broadcast_buffers = False,
                            find_unused_parameters = False)
            self.__model = nn.SyncBatchNorm.convert_sync_batchnorm(self.__model).to(self.__device)
            _optimizer = ZeroRedundancyOptimizer(filter(lambda p: p.requires_grad==True ,self.__model.parameters()),
                                                optimizer_class = Adam,
                                                lr = self.__lr)
        else:
            self.__model.FreezeBackbone()
            _optimizer = Adam(filter(lambda p: p.requires_grad==True ,self.__model.parameters()), lr=self.__lr)
        self.__model = self.__model.to(self.__device)
        #_lrScheduler = LinearLR(_optimizer, total_iters= self.__epochs)
        idx = 0
        self.Evalueate()
        for epoch in range(self.__epochs):
            self.__model.train()
            for batch in tqdm(self.__trainDataLoader, desc='train'):
                batch['inputs'] = {k: v.to(self.__device) for k, v in batch['inputs'].items() if isinstance(v, torch.Tensor)}
                batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
                with amp.autocast():
                    loss = self.__model(**batch)
                if self.__config['synced'] and self.__localRank == 0:
                    self.__Log('train/loss', loss.item(), idx)
                elif not self.__config['synced']:
                    self.__Log('train/loss', loss.item(), idx)
                idx += 1

                if self.__config['synced']:
                    self.__scaler.scale(loss).backward()
                    self.__scaler.step(_optimizer)
                    self.__scaler.update()
                    _optimizer.zero_grad(set_to_none=True)
                    if idx and idx % self.__evalParms['evalStep'] == 0:
                        self.Evalueate()
                        self.__model.train()
                else:
                    self.__scaler.scale(loss).backward()
                    if idx % 1 == 0 or idx + 1 == len(self.__trainDataLoader):
                        self.__scaler.step(_optimizer)
                        self.__scaler.update()
                        _optimizer.zero_grad(set_to_none=True)
                    if idx and idx % self.__evalParms['evalStep'] == 0:
                        self.Evalueate()
                        self.__model.train()        
                #end if synced
            #end for batch
            #_lrScheduler.step()
        #end for epoch
    
    def Evalueate(self):
        self.__model.to(self.__device)
        self.__model.eval()

        lossCollector = []
        with torch.no_grad():
            for batch in self.__valDataLoader:
                batch['inputs'] = {k: v.to(self.__device) for k, v in batch['inputs'].items() if isinstance(v, torch.Tensor)}
                batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
                loss = self.__model(**batch)
                lossCollector.append(loss)
        
        if not self.__config['synced'] or self.__config['synced'] and self.__localRank == 0:
            avgLoss = sum(lossCollector)/len(lossCollector)
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
        self.__model.to(self.__device)
        self.__model.eval()
        seqs = []
        with torch.no_grad():
            for batch in tqdm(self.__testDataLoader, desc='eval'):
                batch['inputs'] = {k: v.to(self.__device) for k, v in batch['inputs'].items() if isinstance(v, torch.Tensor)}
                batch['c_inputs'] = {k: v.to(self.__device) for k, v in batch['c_inputs'].items() if isinstance(v, torch.Tensor)}
                preds = self.__model.inference(**batch)
                seq = self.__tokenizer.decode(preds, skip_special_tokens=True)
                seqs.append(seq)

        fh = open('/data/gj/PromptCompression-vicuna/cprsed_result_414-691.txt', 'w', encoding = 'utf-8')
        for seq in seqs:
            fh.write(seq.replace('\n', '') + '\n')
        fh.close()