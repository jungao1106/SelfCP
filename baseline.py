import torch
import os
import json
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from fastchat.model.model_adapter import get_conversation_template

import torch
from torch.utils.data import Dataset

class DatasetModule(Dataset):
    def __init__(self, data: list, query: str, tokenizer: LlamaTokenizer) -> None:
        super(Dataset, self).__init__()
        self.__data = data
        self.__tokenizer = tokenizer
        self.__query = query
    
        
    def __len__(self) -> int:
        return len(self.__data)
    
    def __getitem__(self, index: int) -> dict:
        conv = get_conversation_template('/data/gj/vicuna-7b-v1.3')
        example = self.__data[index]
        conv.append_message(conv.roles[0], '{}: {}'.format(self.__query, example['dialogue']))
        conv.append_message(conv.roles[1], '')
        
        
        prompt = conv.get_prompt()
        
        
        model_inputs = self.__tokenizer(prompt, return_tensors='pt')
        model_inputs['input_ids'] = model_inputs['input_ids'].flatten()
        model_inputs['attention_mask'] = model_inputs['attention_mask'].flatten()
        return model_inputs

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
model = LlamaForCausalLM.from_pretrained('/data/gj/vicuna-7b-v1.3').half().cuda()
tokenizer = LlamaTokenizer.from_pretrained('/data/gj/vicuna-7b-v1.3')
testData = json.load(open('/data/gj/data/samsum/test.json'))
testDataset = DatasetModule(data = testData,
                            query= 'Summarize the following text',
                            tokenizer = tokenizer,)
testDataloader = DataLoader(dataset=testDataset,
                            batch_size=1,
                            shuffle=False)
seqs = []
model.eval()
for batch in tqdm(testDataloader, desc='vicuna'):
    with torch.no_grad():
        outputs = model.generate(input_ids = batch['input_ids'].to('cuda'),
                                max_length = 2048, num_beams=1,
                                do_sample=True, top_p=0.7, 
                                temperature=0.95, logits_processor=None)
    
    outputs = outputs.tolist()[0][batch['input_ids'][0].shape[0]:]
    seq = tokenizer.decode(outputs)
    seqs.append(seq)
fh = open('/data/gj/PromptCompression-vicuna/vicuna_result.txt', 'w', encoding='utf-8')
for seq in seqs:
    fh.write(seq.replace('\n', '') + '\n')
fh.close()
    