import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.optim import Adam
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from time import time
class ModelModule(nn.Module):
    def __init__(self, modelNameOrPath, options=[], mlpPath=None, embeddingPath=None, normPath=None) -> None:
        super(ModelModule, self).__init__()
        self.__model = AutoModelForCausalLM.from_pretrained(modelNameOrPath, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.__options = options
        for n, p in self.__model.named_parameters():
            p.data = p.data.to(torch.bfloat16)

        if mlpPath:
            self.__mlp = torch.load(mlpPath, map_location='cpu')
        else:
            self.__mlp = nn.Sequential(nn.Linear(4096, 4096))
            for n, p in self.__mlp.named_parameters():
                p.data = p.data.to(torch.bfloat16)

        #self.__mlp = nn.Embedding(1, 4096, dtype=torch.bfloat16)

        if embeddingPath:
            self.__embedding = torch.load(embeddingPath, map_location='cpu')
        else:
            self.__embedding = None
        
                
    def naivecot(self, inputs):
        with torch.no_grad():
            logits = self.__model(inputs['input_ids'])['logits'][:, -1]
            logits = logits[:, self.__options].view(-1, len(self.__options))
            return F.softmax(logits, dim=-1)
        
    def cprsedcot(self, inputs, c_inputs, op):
        with torch.no_grad():
            recInputIds = inputs['input_ids']
            holderEmbeds = None
            for inputIds in recInputIds:
                if op == ['recursive']:
                    oriHiddenStates = self.__model.model(inputIds, holder_embeds=holderEmbeds)[0]
                elif op == ['cat'] or op == ['add']:
                    oriHiddenStates = self.__model.model(inputIds, holder_embeds=None)[0]
                cprsedPostion = (inputIds == 4)
                cprsedHiddenStates = oriHiddenStates[cprsedPostion, ...].view(-1, oriHiddenStates.shape[-1])
                cprsedHiddenStates = self.__mlp(cprsedHiddenStates).to(torch.bfloat16)
                
                if holderEmbeds is None:
                    holderEmbeds = cprsedHiddenStates
                else:
                    if op == ['cat']:
                        holderEmbeds = torch.cat([holderEmbeds, cprsedHiddenStates])
                    elif op == ['recursive']:    
                        holderEmbeds = cprsedHiddenStates
                    elif op ==['add']:
                        holderEmbeds += cprsedHiddenStates

            c_inputs['cprsed_embeds'] = holderEmbeds
            
            logits = self.__model(**c_inputs)['logits'][:, -1]
            logits = logits[:, self.__options].view(-1, len(self.__options))
            return F.softmax(logits, dim=-1)

    def baseline(self, inputs):
        with torch.no_grad():
            decodingTS = time()
            outputs = self.__model.generate(inputs['input_ids'],
                                            min_new_tokens = 32,
                                            max_new_tokens = 256,
                                            repetition_penalty = 1.1,
                                            num_beams = 1)
            outputs = outputs[..., inputs['input_ids'].shape[-1]: ]
            
            
            ''' below used for recursive summarization proposed by open ai'''
            if 'tail' in inputs.keys() and inputs['tail']:
                start = 0
                end = start + 512 - len(outputs[0])
                inputs['prompt'] = [item.item() for item in inputs['prompt']]
                inputs['tail'] = [item.item() for item in inputs['tail']]
                inputs['sepIds'] = [item.item() for item in inputs['sepIds']]
                
                accOuts = outputs.squeeze().tolist()
                while start == 0 or end < len(inputs['tail']):
                    nextIds = [1] + inputs['prompt'] + accOuts + inputs['tail'][start: end] + inputs['sepIds']
                    nextIds = torch.tensor(nextIds, device=inputs['input_ids'].device).unsqueeze(dim=0)
                    outputs = self.__model.generate(nextIds,
                                                    min_new_tokens = 32,
                                                    max_new_tokens = 256,
                                                    num_beams = 1,
                                                    repetition_penalty = 1.1)
                    outputs = outputs[..., nextIds.shape[-1]: ].squeeze().tolist()
                    accOuts += outputs
            
                    start = end
                    end = start + 512 - len(accOuts)
                    
                    if end <= start:
                        break
                    
            if isinstance(outputs, list):
                outputs = torch.tensor(outputs, device = inputs['input_ids'].device).view(1, -1)
            decodingES = time()
            '''end of recursive summarization proposed by open ai'''
            if outputs.shape[-1] > 256:
                return outputs[..., :256], decodingES-decodingTS
            else:
                return F.pad(outputs, [0, 256-outputs.shape[-1]]), decodingES-decodingTS
            
    def inference(self, op, inputs, c_inputs):
        with torch.no_grad():
            encodingTS = time()
            oriHiddenStates = self.__model.model(**inputs)[0]
            cprsed_postion = (inputs['input_ids'] == 4)
            cprsedHiddenStates = oriHiddenStates[cprsed_postion, ...].view(-1, oriHiddenStates.shape[-1])
        
            c_inputs['cprsed_embeds'] = self.__mlp(cprsedHiddenStates).to(torch.bfloat16)
            decodingTS = time()
            outputs = self.__model.generate(**c_inputs,
                                            min_new_tokens = 16,
                                            max_new_tokens = 256, num_beams=1,
                                            repetition_penalty = 1.1)
            outputs = outputs[..., c_inputs['input_ids'].shape[-1]: ]
            decodingES = time()
            if outputs.shape[-1] > 256:
                return outputs[..., :256], decodingES - decodingTS, decodingES - encodingTS
            else:
                return F.pad(outputs, [0, 256-outputs.shape[-1]]), decodingES - decodingTS, decodingES - encodingTS
    
    def forward(self, op, inputs, c_inputs):
        '''
        forward with multi compression method
        1. linear compression math.ceil((len(context)/max_context) * 128)
        2. recursive compression [prompt + context_1 + cprsed, prompt + holder + context_2 + cprsed, .....]
        3. partial compression prompt + partial + context + compressed virtual tokens
        4. concate compression [prompt + context_1 + cprsed + context_2 + cprsed]
        '''
        if op == ['recursive']:
            recInputIds = inputs['input_ids']
            
            cprsedHiddenStates = None
            for inputIds in recInputIds:
                curInputs = {'input_ids': inputIds, 'holder_embeds': cprsedHiddenStates}
                oriHiddenStates = self.__model.model(**curInputs)['last_hidden_state']
                cprsed_postion = (inputIds == 4)
                cprsedHiddenStates = oriHiddenStates[cprsed_postion, ...].view(-1, oriHiddenStates.shape[-1])     
                cprsedHiddenStates = self.__mlp(cprsedHiddenStates).to(torch.bfloat16)
            c_inputs['cprsed_embeds'] = cprsedHiddenStates
            loss = self.__model(**c_inputs)['loss']
            
        elif op == ['concat']:
            recInputIds = inputs['input_ids']
            holderEmbeds = None
            for inputIds in recInputIds:
                curInputs = {'input_ids': inputIds, 'holder_embeds': holderEmbeds}
                oriHiddenStates = self.__model.model(**curInputs)['last_hidden_state']
                cprsed_postion = (inputIds == 4)
                cprsedHiddenStates = oriHiddenStates[cprsed_postion, ...].view(-1, oriHiddenStates.shape[-1])     
                cprsedHiddenStates = self.__mlp(cprsedHiddenStates).to(torch.bfloat16)
                if holderEmbeds is None:
                    holderEmbeds = cprsedHiddenStates
                else:
                    holderEmbeds = torch.cat([holderEmbeds, cprsedHiddenStates])
            c_inputs['cprsed_embeds'] = holderEmbeds
            loss = self.__model(**c_inputs)['loss']
            
        else:
            oriHiddenStates = self.__model.model(**inputs)['last_hidden_state']        
            cprsed_postion = (inputs['input_ids'] == 4)
            cprsedHiddenStates = oriHiddenStates[cprsed_postion, ...].view(-1, oriHiddenStates.shape[-1])
            cprsedHiddenStates = self.__mlp(cprsedHiddenStates).to(torch.bfloat16)
            c_inputs['cprsed_embeds'] = cprsedHiddenStates
            
            loss = self.__model(**c_inputs)['loss']
        return loss
    
    def recursive(self, op, inputs, c_inputs):
        with torch.no_grad():
            recInputIds = inputs['input_ids']
            holderEmbeds = None
            encodingTS = time()
            for inputIds in recInputIds:
                if op == ['recursive']:
                    oriHiddenStates = self.__model.model(inputIds, holder_embeds=holderEmbeds)[0]
                elif op == ['cat'] or op == ['add']:
                    oriHiddenStates = self.__model.model(inputIds, holder_embeds=None)[0]
                cprsedPostion = (inputIds == 4)
                cprsedHiddenStates = oriHiddenStates[cprsedPostion, ...].view(-1, oriHiddenStates.shape[-1])
                cprsedHiddenStates = self.__mlp(cprsedHiddenStates).to(torch.bfloat16)
                
                if holderEmbeds is None:
                    holderEmbeds = cprsedHiddenStates
                else:
                    if op == ['cat']:
                        holderEmbeds = torch.cat([holderEmbeds, cprsedHiddenStates])
                    elif op == ['recursive']:    
                        holderEmbeds = cprsedHiddenStates
                    elif op ==['add']:
                        holderEmbeds += cprsedHiddenStates
                        
            c_inputs['cprsed_embeds'] = holderEmbeds
            decodingTS = time()
            
            outputs = self.__model.generate(**c_inputs,
                                min_new_tokens = 16,
                                max_new_tokens = 128, num_beams=1)
            decodingES = time()
            outputs = outputs[..., c_inputs['input_ids'].shape[-1]: ]
            if outputs.shape[-1] > 128:
                return outputs[..., :128], decodingES - decodingTS, decodingES - encodingTS
            else:
                return F.pad(outputs, [0, 128-outputs.shape[-1]]),  decodingES - decodingTS, decodingES - encodingTS
    
    def retrieve(self, anchorInputs, rerankInputs, k):
        anchorHiddenStates = self.__model.model(anchorInputs)['last_hidden_state']
        cprsed_postion = (anchorInputs == 4)
        anchorHiddenStates = anchorHiddenStates[cprsed_postion, ...].view(-1, anchorHiddenStates.shape[-1])     
        anchorHiddenStates = self.__mlp(anchorHiddenStates).to(torch.bfloat16)
        
        allRankHiddenStates = []
        for inputs in rerankInputs:
            rankHiddenStates = self.__model.model(inputs)['last_hidden_state']
            cprsed_postion = (inputs == 4)
            rankHiddenStates = rankHiddenStates[cprsed_postion, ...].view(-1, rankHiddenStates.shape[-1])     
            rankHiddenStates = self.__mlp(rankHiddenStates).to(torch.bfloat16)
            allRankHiddenStates.append(rankHiddenStates)
        
        if k > 1:
            anchorHiddenStates = anchorHiddenStates[ :k, ...].sum(dim=0).unsqueeze(dim=0)
        
            allRankHiddenStates = torch.stack([item[ :k, ...].sum(dim=0) for item in allRankHiddenStates]).squeeze()
        else:
            anchorHiddenStates = anchorHiddenStates[0].unsqueeze(dim=0)
            allRankHiddenStates = torch.stack([item[0] for item in allRankHiddenStates]).squeeze()
        return anchorHiddenStates @ allRankHiddenStates.T
    
    def encode(self, inputs):
        anchorHiddenStates = self.__model.model(inputs)['last_hidden_state']
        cprsed_postion = (inputs == 4)
        anchorHiddenStates = anchorHiddenStates[cprsed_postion, ...].view(-1, anchorHiddenStates.shape[-1])     
        anchorHiddenStates = self.__mlp(anchorHiddenStates).to(torch.bfloat16)
        return anchorHiddenStates[0]
    
        
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
        
