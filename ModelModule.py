import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.optim import Adam
from transformers import LlamaForCausalLM
from torch.nn import CrossEntropyLoss


class ModelModule(nn.Module):
    def __init__(self, modelNameOrPath, mlpPath=None) -> None:
        super(ModelModule, self).__init__()
        self.__model = LlamaForCausalLM.from_pretrained(modelNameOrPath).half()

        if mlpPath:
            self.__mlp = torch.load(mlpPath)
        else:
            self.__mlp = nn.Linear(4096, 4096)
    
    def inference(self, inputs, c_inputs):
        with torch.no_grad():
            oriHiddenStates = self.__model.model(**inputs)[0]
            cprsed_shape = list(oriHiddenStates.shape)
            cprsed_shape[1] = -1
            cprsed_postion = (inputs['input_ids'] == self.__model.config.pad_token_id) & (inputs['attention_mask'] == 1)
            cprsedHiddenStates = oriHiddenStates[cprsed_postion, ...].view(*cprsed_shape)
            c_inputs['cprsed_embeds'] = self.__mlp(cprsedHiddenStates.float()).half()
            outputs = self.__model.generate(**c_inputs,
                                            max_length = 2048, num_beams=1,
                                            do_sample=True, top_p=0.7, 
                                            temperature=0.95, logits_processor=None)

                
            return outputs.tolist()[0][c_inputs['input_ids'].shape[-1]: ]
            
    
    
    def forward(self, inputs, c_inputs):
        with torch.no_grad():
            oriHiddenStates = self.__model.model(**inputs)[0]       
        
        cprsed_shape = list(oriHiddenStates.shape)
        cprsed_shape[1] = -1
        cprsed_postion = (inputs['input_ids'] == self.__model.config.pad_token_id) & (inputs['attention_mask'] == 1)
        cprsedHiddenStates = oriHiddenStates[cprsed_postion, ...].view(*cprsed_shape)
        
        cprsedHiddenStates = self.__mlp(cprsedHiddenStates.float()).half()
        c_inputs['cprsed_embeds'] = cprsedHiddenStates
        
        loss = self.__model(**c_inputs)['loss']
        return loss

    def FreezeBackbone(self):
        for parm in self.__model.parameters():
                parm.requires_grad = False
            
            
    def SaveTrainedModel(self, val: float ,path: str = './') -> None:
        os.makedirs(path, exist_ok = True)
        torch.save(self.__mlp, os.path.join(path, 'model_loss_{0:.4f}.pth'.format(val)))
    
    def ConfigOptimizer(self, lr: float) -> Adam:
        return Adam(self.parameters(), lr=lr)
    