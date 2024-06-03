import os
import torch
import ruamel.yaml as yaml
import argparse
from DatasetModule import DatasetModule
from ModelModuleLLama import ModelModule
from utils import LoadDatas, GlobalSeed, PromptGather
from torch.utils.data import DataLoader
from Trainer import Trainer
from transformers import LlamaTokenizer, AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
from fastchat.model.model_adapter import get_conversation_template
from calflops import calculate_flops

def main(config):
    initial(config)
    if config['synced']:
        torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizerNameOrPath'], use_fast=False, trust_remote_code=True)
    promptGather = PromptGather([config['sumDescription'], config['qaDescription']])
    system =  get_conversation_template(config['modelNameOrPath']).system
    options = tokenizer.encode('A B C D', add_special_tokens=False, trust_remote_code=True)
    
    compressed_prompt = LoadDatas('/data1/gj/PromptCompression/promptcompressor/promptcompressor_bluelm_duc.json')
    
    #Datasets prepare
    if config['action'] == 'train':
        shortTrainData = LoadDatas(config['shortTrainFile'])
        longTrainData = LoadDatas(config['longTrainFile'])
        trainData =  longTrainData + shortTrainData
        valData = LoadDatas(config['valFile'])
    else:
        testData = LoadDatas(config['testFile'])[:2000]
        for c, t in zip(compressed_prompt, testData):
            t['compressed_prompt'] = c.pop('compressed_prompt')
            
        # print(f'original test size: {len(testData)}')
        # # if task == 'Summarization':
        # testData = [item for item in testData if item['document'] and item['summary']]
        # # elif task == 'QA':
        # #     testData = [item for item in testData if item['Dialogue'] and item['Target'] and item['Question'] and item['Choices'] and item['Human Written Answer']]
        # print(f'filtered test size: {len(testData)}')
    
    
    if config['action'] == 'train':
        valDataset = DatasetModule(data = valData,
                                    system = system,
                                    tokenizer = tokenizer,
                                    maxContextLength = config['maxContextLength'],
                                    maxTargetLength = config['maxTargetLength'],
                                    maxPromptLength = config['maxPromptLength'],
                                    cprsedTokens = config['cprsedTokens'],
                                    action = config['action'],
                                    promptGather = promptGather)
        
        trainDataset = DatasetModule(data = trainData,
                                    system = system,
                                    tokenizer = tokenizer,
                                    maxContextLength = config['maxContextLength'],
                                    maxTargetLength = config['maxTargetLength'],
                                    maxPromptLength = config['maxPromptLength'],
                                    cprsedTokens = config['cprsedTokens'],
                                    action = config['action'],
                                    promptGather = promptGather)
        
    else:
        testDataset = DatasetModule(data = testData,
                                    system = system,
                                    tokenizer = tokenizer,
                                    compression = config['compression'],
                                    task = config['task'],
                                    maxContextLength = config['maxContextLength'],
                                    maxPromptLength = config['maxPromptLength'],
                                    cprsedTokens = config['cprsedTokens'],
                                    action=config['action'],
                                    promptGather = promptGather,
                                    compressedPrompt=compressed_prompt)
    
    #distributed sampler for multi gpus
    if config['action'] == 'train':
        trainSampler = DistributedSampler(trainDataset) if config['synced'] else None
        valSampler = DistributedSampler(valDataset) if config['synced'] else None
    else:
        testSampler = DistributedSampler(testDataset, shuffle=False) if config['synced'] else None
    #integrade into dataloaders
    trainDataLoader, valDataLoader, testDataLoader = None, None, None
    if config['action'] == 'train':
        trainDataLoader = DataLoader(dataset = trainDataset,
                            sampler = trainSampler,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 4,
                            pin_memory = True)
        
        valDataLoader = DataLoader(dataset = valDataset,
                                sampler = valSampler,
                                batch_size = 1,
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
        assert config['mlpPath'] and config['embeddingPath'], 'mlp path or embedding path is None'
        model = ModelModule(config['modelNameOrPath'], config['mlpPath'], config['embeddingPath'])
        
    else: 
        model = ModelModule(config['modelNameOrPath'], options, None if config['action'] == 'train' else config['mlpPath'],
                                                       None if config['action'] == 'train' else config['embeddingPath'],
                                                       None if config['action'] == 'train' else config['normPath'])
        
    #intergrade model dataloader into trainer
    trainer = Trainer(model, [trainDataLoader, valDataLoader, testDataLoader], tokenizer, config['hParams'], config)
    if config['action'] == 'train':
         trainer.Train() #begin train
        
    elif config['action'] == 'baseline':
        if config['task'] == 'context-cot':
            trainer.BaselineCOT()
        else:    
            trainer.Baseline()
        
    elif config['action'] == 'context':
        if 'cot' in config['task']:
            trainer.ContextCOT()
        else:
            trainer.Context()
        
    elif config['action'] == 'qfs':
        trainer.QFS()

    elif config['action'] == 'inference':
        if config['compression'] == 'recursive':
            trainer.Recursive()
        else:
            trainer.Test()
    elif config['action'] == 'computetime':
        trainer.ComputeTime()

def initial(config):
    GlobalSeed(config['seed'])
    if config['synced']:
        #os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA']
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.cuda.empty_cache()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA'][0]
        torch.cuda.set_device(int(config['CUDA'][0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #/data1/gj/PromptCompression/vicuna-1.3/config.yaml
    #/data1/gj/PromptCompression/config-blue.yaml
    parser.add_argument('--config', default='/data1/gj/PromptCompression/vicuna-1.3/config.yaml', help='global environment configs')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(config)
