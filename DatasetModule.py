import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DatasetModule(Dataset):
    def __init__(self, data: list, query: str, tokenizer: AutoTokenizer, 
                 maxSourceLength: int = 512, maxTargetLength: int = 512, 
                 maxCprsedLength: int = 512, maskNums: int = 32) -> None:
        super(Dataset, self).__init__()
        self.__data = data
        self.__tokenizer = tokenizer
        self.__maxSourceLength = maxSourceLength
        self.__maxCprsedLength = maxCprsedLength
        self.__maxTargetLength = maxTargetLength
        self.__maskNums = maskNums
        self.__query = query
        
    def __len__(self) -> int:
        return len(self.__data)
    
    def __getitem__(self, index: int) -> dict:
        example = self.__data[index]
        model_inputs = self.preprocess_function_train(example, self.__query, True)
        
        return model_inputs

    def preprocess_function_train(self, example, query, ignorePadToken):
        maxInputLength = self.__maxSourceLength + self.__maxTargetLength + self.__maskNums
        maxCprsedInputLength = self.__maxCprsedLength + self.__maxTargetLength + self.__maskNums
        
        model_inputs = dict()
        if example['dialogue'] and example['summary']:
            source, target = example['dialogue'], example['summary']

        prompt = "[Round {}]\n问：{}".format(0, query + source)
        c_prompt = "[Round {}]\n问：{}".format(0, query)
        
        a_ids = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = self.__tokenizer.encode(text=target, add_special_tokens=False)
        c_ids = self.__tokenizer.encode(text=c_prompt, add_special_tokens=False)
        mask_ids = self.__tokenizer.encode(text=' '.join(['[MASK]'] * self.__maskNums) + '\n答：', add_special_tokens=False)
        
        
        if len(a_ids) > self.__maxSourceLength - 1 - 2:
            a_ids = a_ids[: self.__maxSourceLength - 1 - 2]

        if len(b_ids) > self.__maxTargetLength - 2:
            b_ids = b_ids[: self.__maxTargetLength - 2]
            
        if len(c_ids) > self.__maxCprsedLength - 1 - 2:
            c_ids = c_ids[: self.__maxCprsedLength - 1 - 2]
                
        a_ids.extend(mask_ids)
        c_ids.extend(mask_ids)
            
        input_ids = self.__tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        
        context_length = input_ids.index(self.__tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position+1:]
        
        pad_len = maxInputLength - len(input_ids)
        input_ids = input_ids + [self.__tokenizer.pad_token_id] * pad_len
        labels = labels + [self.__tokenizer.pad_token_id] * pad_len
        
        
        c_input_ids = self.__tokenizer.build_inputs_with_special_tokens(c_ids, b_ids)
        
        context_length = c_input_ids.index(self.__tokenizer.bos_token_id)
        mask_position = context_length - 1
        c_labels = [-100] * context_length + c_input_ids[mask_position+1:]
        
        pad_len = maxCprsedInputLength - len(c_input_ids)
        c_input_ids = c_input_ids + [self.__tokenizer.pad_token_id] * pad_len
        c_labels = c_labels + [self.__tokenizer.pad_token_id] * pad_len
        
        if ignorePadToken:
            labels = [(l if l != self.__tokenizer.pad_token_id else -100) for l in labels]
            c_labels = [(l if l != self.__tokenizer.pad_token_id else -100) for l in c_labels]

        model_inputs['input_ids'] = torch.tensor(input_ids).flatten()
        model_inputs['c_input_ids'] = torch.tensor(c_input_ids).flatten()
        model_inputs['labels'] = torch.tensor(labels).flatten()
        model_inputs['c_labels'] = torch.tensor(c_labels).flatten()

        return model_inputs
