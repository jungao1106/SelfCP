import os
import torch
import ruamel.yaml as yaml
import argparse
from DatasetModule import DatasetModule
from ModelModule import ModelModule
from utils import LoadDatas, GlobalSeed
from torch.utils.data import DataLoader
from Trainer import Trainer
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler

def main(config):
    initial(config)
    if config['synced']:
        torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count())
    hParams = config['hParams']
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizerNameOrPath'], trust_remote_code=True)
    #Datasets prepare
    valDataset = DatasetModule(data = LoadDatas(config['valFile']),
                               query= 'Summarize the following text: ',
                                tokenizer = tokenizer,
                                maxSourceLength = hParams['maxSourceLength'],
                                maxTargetLength = hParams['maxTargetLength'],
                                maxCprsedLength = hParams['maxCprsedLength'],
                                maskNums = hParams['maskNums'])
    
    trainDataset = DatasetModule(data = LoadDatas(config['trainFile']),
                                 query= 'Summarize the following text: ',
                                 tokenizer = tokenizer,
                                 maxSourceLength = hParams['maxSourceLength'],
                                 maxTargetLength = hParams['maxTargetLength'],
                                maxCprsedLength = hParams['maxCprsedLength'],
                                maskNums = hParams['maskNums'])
    
    #distributed sampler for multi gpus
    trainSampler = DistributedSampler(trainDataset) if config['synced'] else None
    valSampler = DistributedSampler(valDataset) if config['synced'] else None
    #integrade into dataloaders
    trainDataLoader = DataLoader(dataset = trainDataset,
                                 sampler = trainSampler,
                                 batch_size = 1,
                                 shuffle = False,
                                 num_workers = 4,
                                 pin_memory = True)
    
    valDataLoader = DataLoader(dataset = valDataset,
                               sampler = valSampler,
                               batch_size = 4,
                               shuffle = False,
                               num_workers = 4,
                               pin_memory = True)
    #my model
    if config['resume']:
        model = torch.load(config['ckptPath'], map_location='cpu')
        print('reuse checkpoint:' + config['ckptPath'])
    else: 
        model = ModelModule(config['modelNameOrPath'])
        
    #intergrade model dataloader into a trainer
    trainer = Trainer(model, [trainDataLoader, valDataLoader], tokenizer, config['hParams'], config)
    trainer.Train() #begin train

def initial(config):
    GlobalSeed(config['seed'])
    if config['synced']:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA']
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.cuda.empty_cache()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA'][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/data/gj/PromptCompression/config.yaml', help='global environment configs')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(config)
