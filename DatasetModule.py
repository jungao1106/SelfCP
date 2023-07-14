import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer

class DatasetModule(Dataset):
    def __init__(self, data: list, query: str, system: str, tokenizer: LlamaTokenizer, 
                 maxSourceLength: int = 512, maxTargetLength: int = 512, 
                 maxCprsedLength: int = 512, cprsedTokens: int = 32, doTrain: bool = True) -> None:
        super(Dataset, self).__init__()
        self.__data = data
        self.__tokenizer = tokenizer
        self.__maxSourceLength = maxSourceLength
        self.__maxCprsedLength = maxCprsedLength
        self.__maxTargetLength = maxTargetLength
        self.__cprsedTokens = cprsedTokens
        self.__query = query
        self.__system = system
        self.__doTrain = doTrain
        
    def __len__(self) -> int:
        return len(self.__data)
    
    def __getitem__(self, index: int) -> dict:
        example = self.__data[index]
        if self.__doTrain:
            modelInputs = self.PreprocessForTrainWODescription(example, self.__query, True)
            #model_inputs = self.PreprocessForTrain(example, self.__query, True)
        else:
            modelInputs = self.PreprocessForTest(example, self.__query)
            
        return modelInputs

    def PreprocessForTest(self, example, query):
        if example['dialogue'] and example['summary']:
            source = example['dialogue']

        prompt = "{}: {}".format(query, source)
        c_prompt = "{}:".format(query)
        cprsed = '<unk>'*self.__cprsedTokens
        
        sourceIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        c_sourceIds = self.__tokenizer.encode(text=c_prompt, add_special_tokens=False)

        # cprsed_ids = self.__tokenizer.encode(text=cprsed, add_special_tokens=False)
        
        if len(sourceIds) > self.__maxSourceLength:
            prompt = self.__tokenizer.decode(sourceIds[: self.__maxSourceLength], skip_special_tokens=True)
            sourceIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
            
        if len(c_sourceIds) > self.__maxCprsedLength:
            c_prompt = self.__tokenizer.decode(c_sourceIds[: self.__maxCprsedLength], skip_special_tokens=True)
            c_sourceIds = self.__tokenizer.encode(text=c_prompt, add_special_tokens=False)
        inputIds = self.__tokenizer(f'{self.__system} USER: {prompt+cprsed} ASSISTANT:').input_ids
        c_inputIds = self.__tokenizer(f'{self.__system} USER: {c_prompt+cprsed} ASSISTANT:').input_ids      

        inputAttentionMask = [1]*len(inputIds)
        c_inputAttentionMask = [1]*len(c_inputIds)      
        
        # pad_len = self.__maxSourceLength - len(source_ids)
        # input_ids = input_ids + [self.__tokenizer.pad_token_id] * pad_len
        # input_attention_mask = input_attention_mask + [0] * pad_len
        
        
        # pad_len = self.__maxCprsedLength - len(c_source_ids)
        # c_input_ids = c_input_ids + [self.__tokenizer.pad_token_id] * pad_len
        # c_input_attention_mask = c_input_attention_mask + [0] * pad_len
        
        return {
            'inputs':{
                'input_ids': torch.tensor(inputIds).flatten(),
                'attention_mask': torch.tensor(inputAttentionMask).flatten()
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten(),
                'attention_mask': torch.tensor(c_inputAttentionMask).flatten(),
            }
        }


    def PreprocessForTrain(self, example, query, ignorePadToken):
        if example['dialogue'] and example['summary']:
            source, target = example['dialogue'], example['summary']

        prompt = "{}: {}".format(query, source)
        c_prompt = "{}:".format(query)
        cprsed = '<unk>'*self.__cprsedTokens
        
        sourceIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        targetIds = self.__tokenizer.encode(text=target, add_special_tokens=False)
        c_sourceIds = self.__tokenizer.encode(text=c_prompt, add_special_tokens=False)
        
        if len(sourceIds) > self.__maxSourceLength:
            prompt = self.__tokenizer.decode(sourceIds[: self.__maxSourceLength], skip_special_tokens=True)
            sourceIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
            
        if len(targetIds) > self.__maxTargetLength:
            target = self.__tokenizer.decode(targetIds[: self.__maxTargetLength], skip_special_tokens=True)
            targetIds = self.__tokenizer.encode(text=target, add_special_tokens=False)
            
        if len(c_sourceIds) > self.__maxCprsedLength:
            c_prompt = self.__tokenizer.decode(c_sourceIds[: self.__maxCprsedLength], skip_special_tokens=True)
            c_sourceIds = self.__tokenizer.encode(text=c_prompt, add_special_tokens=False)
            
        inputIds = self.__tokenizer(f'{self.__system} USER: {prompt+cprsed} ASSISTANT: </s>').input_ids
        c_inputIds = self.__tokenizer(f'{self.__system} USER: {c_prompt+cprsed} ASSISTANT: {target}</s>').input_ids
         
        inputAttentionMask = [1]*len(inputIds)
        c_inputAttentionMask = [1]*len(c_inputIds)      
        
        padLen = self.__maxSourceLength - len(sourceIds)
        inputIds = inputIds + [self.__tokenizer.pad_token_id] * padLen
        inputAttentionMask = inputAttentionMask + [0] * padLen
        
        # 5 is the number ids of 'ASSISTANT:'
        # 1 is the space token after <unk>
        c_context_length = c_inputIds.index(self.__tokenizer.pad_token_id) + self.__maskNums + 5 + 1
        c_labels = [-100] * c_context_length + c_inputIds[c_context_length:]
        
        padLen = self.__maxCprsedLength + self.__maxTargetLength - len(c_sourceIds) - len(targetIds)
        c_inputIds = c_inputIds + [self.__tokenizer.pad_token_id] * padLen
        c_labels = c_labels + [self.__tokenizer.pad_token_id] * padLen
        c_inputAttentionMask = c_inputAttentionMask + [0] * padLen
        
        if ignorePadToken:
            c_labels = [(l if l != self.__tokenizer.pad_token_id or c_inputAttentionMask[i] else -100) for i, l in enumerate(c_labels)]


        return {
            'inputs':{
                'input_ids': torch.tensor(inputIds).flatten(),
                'attention_mask': torch.tensor(inputAttentionMask).flatten()
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten(),
                'attention_mask': torch.tensor(c_inputAttentionMask).flatten(),
                'labels': torch.tensor(c_labels).flatten()
            }
        }
        
        
    def PreprocessForTrainWODescription(self, example, query, ignorePadToken):
        if example['dialogue'] and example['summary']:
            source, target = example['dialogue'], example['summary']

        prompt = source
        c_prompt = "{}:".format(query)
        cprsed = '<unk>'*self.__cprsedTokens
        
        sourceIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        targetIds = self.__tokenizer.encode(text=target, add_special_tokens=False)
        c_sourceIds = self.__tokenizer.encode(text=c_prompt, add_special_tokens=False)
        
        if len(sourceIds) > self.__maxSourceLength:
            prompt = self.__tokenizer.decode(sourceIds[: self.__maxSourceLength], skip_special_tokens=True)
            sourceIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
            
        if len(targetIds) > self.__maxTargetLength:
            target = self.__tokenizer.decode(targetIds[: self.__maxTargetLength], skip_special_tokens=True)
            targetIds = self.__tokenizer.encode(text=target, add_special_tokens=False)
            
        if len(c_sourceIds) > self.__maxCprsedLength:
            c_prompt = self.__tokenizer.decode(c_sourceIds[: self.__maxCprsedLength], skip_special_tokens=True)
            c_sourceIds = self.__tokenizer.encode(text=c_prompt, add_special_tokens=False)
            
        inputIds = self.__tokenizer(f'{prompt+cprsed}').input_ids
        c_inputIds = self.__tokenizer(f'{self.__system} USER: {c_prompt+cprsed} ASSISTANT: {target}</s>').input_ids
         
        inputAttentionMask = [1]*len(inputIds)
        c_inputAttentionMask = [1]*len(c_inputIds)      
        
        padLen = self.__maxSourceLength - len(sourceIds)
        inputIds = inputIds + [self.__tokenizer.pad_token_id] * padLen
        inputAttentionMask = inputAttentionMask + [0] * padLen
        
        # 5 is the number ids of 'ASSISTANT:'
        # 1 is the space token after <unk>
        c_context_length = c_inputIds.index(self.__tokenizer.pad_token_id) + self.__maskNums + 5 + 1
        c_labels = [-100] * c_context_length + c_inputIds[c_context_length:]
        
        padLen = self.__maxCprsedLength + self.__maxTargetLength - len(c_sourceIds) - len(targetIds)
        c_inputIds = c_inputIds + [self.__tokenizer.pad_token_id] * padLen
        c_labels = c_labels + [self.__tokenizer.pad_token_id] * padLen
        c_inputAttentionMask = c_inputAttentionMask + [0] * padLen
        
        if ignorePadToken:
            c_labels = [(l if l != self.__tokenizer.pad_token_id or c_inputAttentionMask[i] else -100) for i, l in enumerate(c_labels)]


        return {
            'inputs':{
                'input_ids': torch.tensor(inputIds).flatten(),
                'attention_mask': torch.tensor(inputAttentionMask).flatten()
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten(),
                'attention_mask': torch.tensor(c_inputAttentionMask).flatten(),
                'labels': torch.tensor(c_labels).flatten()
            }
        }
