import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.optim import Adam
from transformers import LlamaForCausalLM
from time import time

class ModelModule(nn.Module):
    def __init__(self, modelNameOrPath, options=[], mlpPath=None, embeddingPath=None, normPath=None) -> None:
        super(ModelModule, self).__init__()
        self.__model = LlamaForCausalLM.from_pretrained(modelNameOrPath)
    
        for n, p in self.__model.named_parameters():
            p.data = p.data.to(torch.bfloat16)

        if mlpPath:
            self.__mlp = torch.load(mlpPath, map_location='cpu')
        else:
            self.__mlp = nn.Sequential(nn.Linear(4096, 4096))
            for n, p in self.__mlp.named_parameters():
                p.data = p.data.to(torch.bfloat16)

        if embeddingPath:
            self.__embedding = torch.load(embeddingPath, map_location='cpu')
        else:
            self.__embedding = None
            
        if normPath:
            self.__norm = torch.load(normPath, map_location='cpu')
        else:
            self.__norm = nn.LayerNorm(4096)
            for n, p in self.__norm.named_parameters():
                p.data = p.data.to(torch.bfloat16)
        
        self.__options = options
        

    def baseline(self, inputs):
        with torch.no_grad():
            logits = self.__model(**inputs)['logits']
            opts = F.softmax(logits[torch.arange(logits.shape[-1]) == self.__options], dim=-1)
            return opts
        
    def cot(self, inputs, c_inputs):
        with torch.no_grad():
            oriHiddenStates =self.__model.model(**inputs)
            cprsed_postion = (inputs['input_ids'] == 4)
            cprsedHiddenStates = oriHiddenStates[cprsed_postion, ...].view(-1, oriHiddenStates.shape[-1])
        
   
    def SetTrainedEmbedding(self):
        assert self.__model.model.frozen_embeddings is not None, 'Please separate initial embeddings first'
        assert self.__embedding is not None, 'Please load trained embeddings during inference'
        self.__model.model.set_trained_embeddings(self.__embedding)
    
    def SeperateEmbedding(self):
        self.__model.model.seperate_embeddings()
    
    def FreezeBackbone(self):
        for parm in self.__model.parameters():
                parm.requires_grad = False
            
    def SaveTrainedModel(self, val: float ,path: str = './') -> None:
        os.makedirs(path, exist_ok = True)
        torch.save(self.__mlp, os.path.join(path, 'model_loss_{0:.4f}.pth'.format(val)))
        torch.save(self.__model.model.trained_embeddings, os.path.join(path, 'embedding_loss_{0:.4f}.pth'.format(val)))
        torch.save(self.__norm, os.path.join(path, 'norm_loss_{0:.4f}.pth'.format(val)))