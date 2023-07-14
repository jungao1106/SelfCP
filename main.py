import os
import torch
import ruamel.yaml as yaml
import argparse
from DatasetModule import DatasetModule
from ModelModule import ModelModule
from utils import LoadDatas, GlobalSeed
from torch.utils.data import DataLoader
from Trainer import Trainer
from transformers import LlamaTokenizer
from torch.utils.data.distributed import DistributedSampler
from fastchat.model.model_adapter import get_conversation_template

def main(config):
    initial(config)
    if config['synced']:
        torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count())
    hParams = config['hParams']
    tokenizer = LlamaTokenizer.from_pretrained(config['tokenizerNameOrPath'])
    system = get_conversation_template(config['modelNameOrPath']).system
    
    #Datasets prepare
    if config['doTrain']:
        trainData = LoadDatas(config['trainFile'])
        print(f'original train size: {len(trainData)}')
        trainData = [item for item in trainData if item['dialogue'] and item['summary']]
        print(f'filtered train size: {len(trainData)}')
        valData = LoadDatas(config['valFile'])
        print(f'original val size: {len(valData)}')
        valData = [item for item in valData if item['dialogue'] and item['summary']]
        print(f'filtered val size: {len(valData)}')
    else:
        testData = LoadDatas(config['testFile'])
        print(f'original test size: {len(testData)}')
        testData = [testData[413], testData[690]]
        testData = [item for item in testData if item['dialogue'] and item['summary']]
        print(f'filtered test size: {len(testData)}')
    
    if config['doTrain']:
        valDataset = DatasetModule(data = valData,
                                    query= 'Summarize the following text',
                                    system=system,
                                    tokenizer = tokenizer,
                                    maxSourceLength = hParams['maxSourceLength'],
                                    maxTargetLength = hParams['maxTargetLength'],
                                    maxCprsedLength = hParams['maxCprsedLength'],
                                    cprsedTokens = hParams['cprsedTokens'],
                                    doTrain = config['doTrain'])
        
        trainDataset = DatasetModule(data = trainData,
                                    query= 'Summarize the following text',
                                    system=system,
                                    tokenizer = tokenizer,
                                    maxSourceLength = hParams['maxSourceLength'],
                                    maxTargetLength = hParams['maxTargetLength'],
                                    maxCprsedLength = hParams['maxCprsedLength'],
                                    cprsedTokens = hParams['cprsedTokens'],
                                    doTrain = config['doTrain'])
    else:
        testDataset = DatasetModule(data = testData,
                                    query= 'Summarize the following text',
                                    system=system,
                                    tokenizer = tokenizer,
                                    maxSourceLength = hParams['maxSourceLength'],
                                    maxCprsedLength = hParams['maxCprsedLength'],
                                    cprsedTokens = hParams['cprsedTokens'],
                                    doTrain=config['doTrain'])
    
    #distributed sampler for multi gpus
    if config['doTrain']:
        trainSampler = DistributedSampler(trainDataset) if config['synced'] else None
        valSampler = DistributedSampler(valDataset) if config['synced'] else None
    else:
        testSampler = DistributedSampler(testDataset) if config['synced'] else None
    
    #integrade into dataloaders
    trainDataLoader, valDataLoader, testDataLoader = None, None, None
    if config['doTrain']:
        trainDataLoader = DataLoader(dataset = trainDataset,
                                    sampler = trainSampler,
                                    batch_size = 3,
                                    shuffle = False,
                                    num_workers = 4,
                                    pin_memory = True)
        
        valDataLoader = DataLoader(dataset = valDataset,
                                sampler = valSampler,
                                batch_size = 3,
                                shuffle = False,
                                num_workers = 4,
                                pin_memory = True)
    else:
        testDataLoader = DataLoader(dataset = testDataset,
                                sampler = testSampler,
                                batch_size = 1,
                                shuffle = False,
                                pin_memory = True)
    #model
    if config['resume']:
        model = torch.load(config['ckptPath'], map_location='cpu')
        print('reuse checkpoint:' + config['ckptPath'])
    else: 
        model = ModelModule(config['modelNameOrPath'], None if config['doTrain'] else config['mlpPath'])
        
    #intergrade model dataloader into trainer
    trainer = Trainer(model, [trainDataLoader, valDataLoader, testDataLoader], tokenizer, config['hParams'], config)
    if config['doTrain']:
        trainer.Train() #begin train
    else:
        trainer.Test()

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
    parser.add_argument('--config', default='/data/gj/PromptCompression-vicuna/config.yaml', help='global environment configs')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(config)
