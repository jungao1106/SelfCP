import torch
import math
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from utils import PromptGather
from numpy.random import uniform
class DatasetModule(Dataset):
    def __init__(self, data: list, system: str, tokenizer: LlamaTokenizer, compression: str = None, task: str = None,
                 maxContextLength: int = 512, maxTargetLength: int = 512, maxPromptLength: int = 512, 
                 cprsedTokens: int = 32, action: str = 'train', promptGather: PromptGather = None,
                 compressedPrompt = []) -> None:
        super(Dataset, self).__init__()
        self.__data = data
        self.__tokenizer = tokenizer
        self.__maxContextLength = maxContextLength
        self.__maxPromptLength = maxPromptLength
        self.__maxTargetLength = maxTargetLength
        self.__cprsedTokens = cprsedTokens
        self.__system = system
        self.__action = action
        self.__promptGather = promptGather
        self.__compression = compression
        self.__task = task
        self.__compressedPrompt = compressedPrompt
        self.__ratio = 12
        
    def __len__(self) -> int:
        return len(self.__data)
    
    def __getitem__(self, index: int) -> dict:
        example = self.__data[index]
        # if 'cot' in self.__task:
        #     example['similar0'] = self.__data[(index+1)%len(self.__data)]
        #     example['similar1'] = self.__data[(index+2)%len(self.__data)]
        if self.__action == 'train':
            #modelInputs = self.PreprocessForTrain(example, True)
            modelInputs = self.PreprocessForMultiTrain(example, True)
            
        elif self.__action == 'baseline':
            compressed_prompt = None
            if len(self.__compressedPrompt) == 2:
                compressed_prompt = self.__compressedPrompt[0]['compressed_prompt'] #+ self.__compressedPrompt[1]['compressed_prompt']
            else:
                compressed_prompt = example['compressed_prompt']
            modelInputs = self.PreprocessForBaseline(example, self.__compression, self.__task, 1, compressed_prompt)
        
        elif self.__action == 'context':
            if 'cot' in self.__task:
                modelInputs = self.PreprocessForCprsedCOT(example, self.__compression, self.__task, 2)
            else:
                modelInputs = self.PreprocessForInContextTest(example, self.__compression ,self.__task, 1)
                
        elif self.__action == 'qfs':
                modelInputs = self.PreprocessForQFSTest(example, self.__compression, self.__task)  
             
        elif self.__action == 'inference':
            if self.__compression == 'recursive':
                modelInputs = self.PreprocessForRecursiveTest(example) 
            elif self.__compression == 'partial':
                modelInputs = self.PreprocessForPartialTest(example, self.__task)
            else:
                modelInputs = self.PreprocessForTest(example, self.__compression, self.__task)
                
        return modelInputs

    def SetRatio(self, r):
        self.__ratio = r

    def PreprocessForNaiveCOT(self, example, task, fewshot):
        example['task'] = task
        example['fewshot'] = fewshot
        prompt, contexts = self.__promptGather.ConstructBaselinePrompts(example)
        promptIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)[:self.__maxPromptLength]
        contextIds = [self.__tokenizer.encode(text=context, add_special_tokens=False) for context in contexts]
        if fewshot > 0:
            contextIds = contextIds[0] if fewshot == 1 else contextIds[0] + contextIds[1]
            inputIds = contextIds[ :self.__maxContextLength - len(promptIds)] + promptIds
        else:
            inputIds = promptIds[ :self.__maxContextLength]
        
        return {
            'inputs': {
                'input_ids': torch.tensor(inputIds).flatten()
            }
        }        
    
    def PreprocessForCprsedCOT(self, example, compression, task, fewshot):
        example['fewshot'] = fewshot
        example['task'] = task
        prompt, contexts, _ = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()
        
        fewContextIds = [self.__tokenizer.encode(text=context, add_special_tokens=False) for context in contexts]
        fewContextIds = [contextIds[ :self.__maxContextLength] for contextIds in fewContextIds]
        
        promptIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        promptIds = promptIds[ :self.__maxPromptLength]

        cprsed_nums = self.__cprsedTokens
        cprsed_nums = math.ceil(sum([len(item) for item in fewContextIds])/len(fewContextIds)/12)
        cprsed = [4] * cprsed_nums
        inputIdsHead = self.__tokenizer(f'{self.__system}').input_ids

        fewInputIds = []
        for i in range(fewshot):
            fewInputIds.append(inputIdsHead + [3] +  fewContextIds[i] + cprsed)
            
        c_inputIdsHead = self.__tokenizer(f'{self.__system} USER:').input_ids
        c_inputIdsTail = self.__tokenizer(f'ASSISTANT: The Answer is', add_special_tokens=False).input_ids
    
        if compression == 'cat':
            c_inputIds = c_inputIdsHead + cprsed*len(fewInputIds) + promptIds + c_inputIdsTail
        else:
            c_inputIds = c_inputIdsHead + cprsed + promptIds + c_inputIdsTail
        return {
            'op': compression,
            'inputs':{
                'input_ids': [torch.tensor(inputIds).flatten() for inputIds in fewInputIds],
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten()
            }
        }
        
    def PreprocessForBaseline(self, example, compression, task, fewshot=0, compressedPrompt = None):
            example['task'] = task
            example['fewshot'] = fewshot
            prompt, context = self.__promptGather.ConstructBaselinePrompts(example)
            prompt = prompt.strip()
                
            promptIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)[ :self.__maxPromptLength]
            contextIds = self.__tokenizer.encode(text=context[0] if isinstance(context, list) else context, add_special_tokens=False)
            #sepIds = self.__tokenizer.encode('\n根据上述案情,判决书如下：', add_special_tokens=False)
            sepIds = self.__tokenizer.encode('\nThe summary of the above document is:', add_special_tokens=False)

            contextIds = contextIds[: self.__maxContextLength]
            contextIds = contextIds[:len(contextIds)//2]    
            tail = contextIds[self.__maxContextLength: ]
            #contextIds = contextIds[: self.__maxContextLength]
            if 'context' in task:
                
                #zero
                #inputIds = [self.__tokenizer.bos_token_id] + promptIds + sepIds
                # #one
                inputIds = [self.__tokenizer.bos_token_id] + contextIds[ :self.__maxContextLength] + promptIds + sepIds
                #inputIds = inputIdsHead + contextIds[ :max(0, self.__maxContextLength-promptLen)] + self.__tokenizer(f'USER:', add_special_tokens=False).input_ids + promptIds + inputIdsTail
            else:    
                if compressedPrompt is None:
                    #inputIds = inputIdsHead + promptIds + contextIds + inputIdsTail
                    inputIds = [self.__tokenizer.bos_token_id] + promptIds + contextIds + sepIds
                else:
                    compressedPromptIds = self.__tokenizer.encode(compressedPrompt, add_special_tokens=False)
                    inputIds = [self.__tokenizer.bos_token_id] + promptIds + contextIds + compressedPromptIds+ sepIds
            
            if compression == 'recursive':
                return {
                    'inputs': {
                        'input_ids': torch.tensor(inputIds).flatten(),
                        'prompt': promptIds,
                        'tail': tail,
                        'sepIds': sepIds
                    }}
            else:
                return {
                    'inputs': {
                        'input_ids': torch.tensor(inputIds).flatten(),
                    }}   
    
    def PreprocessForPartialTest(self, example, task):
        '''Assuming the max input length is 512'''
        #example['task'] = task
        prompt, context, _ = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()

        contextIds = self.__tokenizer.encode(text=context, add_special_tokens=False)[:self.__maxContextLength]
        promptIds = self.__tokenizer.encode(text=prompt)[ :self.__maxPromptLength]
        sepIds = self.__tokenizer.encode('\nThe summary of the above document is:', add_special_tokens=False)

        #sepIds = self.__tokenizer.encode('\n根据上述案情,判决书如下：', add_special_tokens=False)
        # if example['task'] == 'sum':
        #     sepIds = self.__tokenizer.encode('\nThe summary of the above document is:', add_special_tokens=False)
        # elif example['task'] == 'qa':
        #     sepIds = self.__tokenizer.encode('\nThe answer is:', add_special_tokens=False)
        p = math.ceil(0.5*len(contextIds))
        
        ratio = self.__ratio
        cprsed_nums = math.ceil(p // ratio)
        
        cprsed = [4] * cprsed_nums
        #compress former
        inputIds = promptIds + [3] + contextIds[:p] + cprsed
        c_inputIds = promptIds + cprsed + contextIds[p:] + sepIds
        
        # compress latter
        # inputIds = promptIds + [3] + contextIds[p: ] + cprsed
        # c_inputIds = promptIds + contextIds[:p] + cprsed + sepIds
  
        return {
            'op': 'partial',
            'inputs':{
                'input_ids': torch.tensor(inputIds).flatten(),
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten(),
            }}
          
    def PreprocessForTest(self, example, compression, task):
        example['task'] = task
        prompt, context, _ = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()
        
        promptIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        contextIds = self.__tokenizer.encode(text=context, add_special_tokens=False)
        sepIds = self.__tokenizer.encode(text='\nThe answer is:', add_special_tokens=False)
        
        promptIds = promptIds[ :self.__maxPromptLength]
        contextIds = contextIds[ :self.__maxContextLength]

        if compression == 'linear':
            cprsed_nums = math.ceil(len(contextIds)/12)#math.ceil(self.__cprsedTokens * (len(contextIds) / self.__maxContextLength))
        elif compression == 'fix':
            cprsed_nums = self.__cprsedTokens
        
        cprsed = [4]* cprsed_nums
        
        # if task == 'qa':
        #     inputIds = [self.__tokenizer.bos_token_id] + [3] + contextIds + promptIds + cprsed
            
        #     c_inputIds = [self.__tokenizer.bos_token_id] + cprsed + promptIds + sepIds
        # else:
        inputIds = [self.__tokenizer.bos_token_id] + [3] + promptIds[:len(promptIds)//2] + contextIds + cprsed 

        c_inputIds = [self.__tokenizer.bos_token_id] + promptIds + cprsed + sepIds

        return {
            'op': 'inference',
            'inputs':{
                'input_ids': torch.tensor(inputIds).flatten(),
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten(),
            }
        }

    def PreprocessForMultiTrain(self, example, ignorePadToken):
        prompt, context, target = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()
        
        promptIds = self.__tokenizer.encode(text=prompt)
        contextIds = self.__tokenizer.encode(text=context, add_special_tokens=False)
        targetIds = self.__tokenizer.encode(text=target+self.__tokenizer.eos_token, add_special_tokens=False)
        sepIds = self.__tokenizer.encode('->', add_special_tokens=False)
        
        promptIds = promptIds[ :self.__maxPromptLength]    
        contextIds = contextIds[ :self.__maxContextLength]
        targetIds = targetIds[ :self.__maxTargetLength]

        drop = uniform()
        partialLength = 0
        if 0 <= drop < 0.5:
            op = 'partial'
            if len(contextIds) < 768:
                partialLength = math.ceil(0.5*len(contextIds))
                cprsed_nums = math.ceil(0.5*len(contextIds)) // 12
                cprsed = [4] * cprsed_nums
                if drop < 0.25: # compress latter
                    inputIds = promptIds + [3] + contextIds[math.ceil(0.5*len(contextIds)): ] + cprsed
                    c_inputIds = promptIds + contextIds[ :math.ceil(0.5*len(contextIds))] + cprsed + sepIds + targetIds
                else: # compress former
                    inputIds = promptIds + [3] + contextIds[ :math.ceil(0.5*len(contextIds))] + cprsed
                    c_inputIds = promptIds + cprsed + contextIds[math.ceil(0.5*len(contextIds)): ] + sepIds + targetIds
                    
            else:
                '''long partial compression'''
                if drop < 0.25: # compress latter
                    partialLength = 512
                    cprsed_nums = len(contextIds[512: ]) // 12
                    cprsed = [4] * cprsed_nums
                    inputIds = promptIds + [3] + contextIds[512: ] + cprsed
                    c_inputIds = promptIds + contextIds[ :512] + cprsed + sepIds + targetIds
                else: # compress former
                    partialLength = len(contextIds[512: ])
                    cprsed_nums = len(contextIds[ :512]) // 12
                    cprsed = [4] * cprsed_nums
                    inputIds = promptIds + [3] + contextIds[ :512] + cprsed
                    c_inputIds = promptIds + cprsed + contextIds[512: ] + sepIds + targetIds

        elif 0.5 <= drop < 0.7:
            op = 'recursive'
            inputIds = []
            step_length = min(512, len(contextIds)//2) + 1
            cprsed_nums = step_length // 12
            holder = [5] * cprsed_nums
            cprsed = [4] * cprsed_nums
            for i in range(0, len(contextIds), step_length):
                inputIds.append(promptIds + holder * min(i, 1) + [3] + contextIds[i: i + step_length] + cprsed)
            c_inputIds = promptIds + cprsed + sepIds +targetIds
            
        elif 0.7 <= drop < 0.8:
            op = 'concat'
            inputIds = []
            step_length = min(512, len(contextIds)//2)
            cprsed_nums = step_length // 12
            holder = [5] * cprsed_nums
            cprsed = [4] * cprsed_nums
            for i in range(0, len(contextIds), step_length):
                inputIds.append(promptIds + holder * len(inputIds) + [3] + contextIds[i: i+step_length] + cprsed)
            c_inputIds = promptIds + cprsed * len(inputIds) + sepIds + targetIds
            
        elif 0.8 <= drop <1:
            op = 'linear'
            cprsed_nums = len(contextIds)//12
            cprsed = [4] * cprsed_nums
            inputIds =  promptIds + [3] + contextIds + cprsed 
            c_inputIds =  promptIds + cprsed + sepIds + targetIds
        
        if op == 'concat':
            c_contextLength = len(promptIds + cprsed*len(inputIds) + sepIds)
        else:
            c_contextLength = len(promptIds + cprsed + sepIds) + partialLength
            
        c_labels = [-100] * c_contextLength + c_inputIds[c_contextLength:]
        
    
        if op == 'recursive' or op == 'concat':
            return {
                'op': op,
                'inputs':{
                    'input_ids': [torch.tensor(item).flatten() for item in inputIds],
                },
                'c_inputs':{
                    'input_ids': torch.tensor(c_inputIds).flatten(),
                    'labels': torch.tensor(c_labels).flatten()
                }
            }
        else:
            return {
                'op': op,
                'inputs': {
                    'input_ids': torch.tensor(inputIds).flatten(),
                },
                'c_inputs': {
                    'input_ids': torch.tensor(c_inputIds).flatten(),
                    'labels': torch.tensor(c_labels).flatten()
                }
            }
        
    def PreprocessForTrain(self, example, ignorePadToken):
        prompt, context, target = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()
        
        promptIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        contextIds = self.__tokenizer.encode(text=context, add_special_tokens=False)
        targetIds = self.__tokenizer.encode(text=target+self.__tokenizer.eos_token, add_special_tokens=False)
        
        if len(contextIds) > self.__maxContextLength:
            contextIds = contextIds[ :self.__maxContextLength]
        
        if len(targetIds) > self.__maxTargetLength:
            targetIds = targetIds[ :self.__maxTargetLength]
            
        if len(promptIds) > self.__maxPromptLength:
            promptIds = promptIds[ :self.__maxPromptLength]

        padLen = self.__maxContextLength + self.__maxPromptLength + self.__cprsedTokens - len(contextIds) - len(promptIds)
        c_padLen = self.__maxPromptLength + self.__maxTargetLength + self.__cprsedTokens - len(promptIds) - len(targetIds) + math.ceil(0.1*self.__maxContextLength)

        # padLen = c_padLen = 0
        
        drop = uniform()
        if drop < 0.3:
            c_padLen = c_padLen - math.ceil(0.1*len(contextIds))
            promptIds = promptIds + contextIds[: math.ceil(0.1*len(contextIds))]
            contextIds = contextIds[math.ceil(0.1*len(contextIds)): ]
    
        cprsed_nums = math.ceil(self.__cprsedTokens * (len(contextIds) / self.__maxContextLength))
        cprsed = [4] * cprsed_nums
        padLen = padLen - cprsed_nums
        c_padLen = c_padLen - cprsed_nums
                
        inputIdsHead = c_inputIdsHead = self.__tokenizer(f'{self.__system} Human:').input_ids
        inputIdsTail = c_inputIdsTail = self.__tokenizer(f'ASSISTANT:', add_special_tokens=False).input_ids

        inputIds = inputIdsHead + promptIds + [3] + contextIds + cprsed + inputIdsTail
        c_inputIds = c_inputIdsHead + promptIds + cprsed + c_inputIdsTail + targetIds
        
        inputAttentionMask = [1]*len(inputIds)
        c_inputAttentionMask = [1]*len(c_inputIds)      
        
        inputIds = inputIds + [self.__tokenizer.pad_token_id] * padLen
        inputAttentionMask = inputAttentionMask + [0] * padLen
        
        c_contextLength = len(c_inputIdsHead + cprsed + c_inputIdsTail)
        c_labels = [-100] * c_contextLength + c_inputIds[c_contextLength:]
        
        c_inputIds = c_inputIds + [self.__tokenizer.pad_token_id] * c_padLen
        c_labels = c_labels + [self.__tokenizer.pad_token_id] * c_padLen
        c_inputAttentionMask = c_inputAttentionMask + [0] * c_padLen
        
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
    
    def PreprocessForRecursiveTest(self, example, task):
        example['task'] = task
        prompt, context, _ = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()
        
        contextIds = self.__tokenizer.encode(text=context, add_special_tokens=False)
        p_contextIds = contextIds[ :156]
        contextIds = contextIds[156: ]
        contextRecIds = [contextIds[i: i+self.__maxContextLength] for i in range(0, len(contextIds), self.__maxContextLength)]

        cprsed_nums = self.__cprsedTokens
        cprsed = [4] * cprsed_nums
        holder = [5] * cprsed_nums
        inputIdsHead = c_inputIdsHead = self.__tokenizer(f'{self.__system} USER: {prompt}').input_ids
        inputIdsTail = c_inputIdsTail = self.__tokenizer(f'ASSISTANT:', add_special_tokens=False).input_ids
        
        inputIds = [inputIdsHead + p_contextIds + [3] + holder * min(i, 1) + contextIds + cprsed + inputIdsTail for i, contextIds in enumerate(contextRecIds)]
        inputIds = inputIds[:1]
        inputIds[-1] = inputIds[-1] + (len(inputIds[0]) - len(inputIds[-1]))*[0]
        c_inputIds = c_inputIdsHead + p_contextIds + cprsed + c_inputIdsTail
        
        return {
            'inputs':{
                'input_ids': [torch.tensor(item) for item in inputIds],
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten()
            }
        }

    def PreprocessForInContextTest(self, example, compression, task, fewshot):
        '''prompt contains the inference example'''
        '''max context(prompt) length should be 768'''
        example['fewshot'] = fewshot
        example['task'] = task
        prompt, contexts, _ = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()
        
        fewContextIds = [self.__tokenizer.encode(text=context, add_special_tokens=False) for context in contexts]
        fewContextIds = [contextIds[ :self.__maxContextLength] for contextIds in fewContextIds]
        
        promptIds = self.__tokenizer.encode(text=prompt, add_special_tokens=False)
        promptIds = promptIds[ :self.__maxPromptLength]

        sepIds = self.__tokenizer.encode('\nThe summary of the above document is:', add_special_tokens=False)
        cprsed_nums = self.__cprsedTokens
        cprsed_nums = math.ceil(sum([len(item) for item in fewContextIds])/len(fewContextIds)/12)
        cprsed = [4] * cprsed_nums
        holder = [5] * cprsed_nums

        fewInputIds = []
        fewInputIds.append([self.__tokenizer.bos_token_id] + [3] +  fewContextIds[0] + cprsed)
        if compression == 'recursive':
            for i in range(1, fewshot):
                fewInputIds.append([self.__tokenizer.bos_token_id] + holder + [3] + fewContextIds[i] + cprsed)
        elif compression == 'cat' or compression == 'add':
            for i in range(1, fewshot):
                fewInputIds.append([self.__tokenizer.bos_token_id] + [3] + fewContextIds[i] + cprsed)
    
        if compression == 'cat':
            c_inputIds = [self.__tokenizer.bos_token_id] + cprsed*len(fewInputIds) + promptIds + sepIds
        else:
            c_inputIds = [self.__tokenizer.bos_token_id] + cprsed + promptIds + sepIds
            
        return {
            'op': compression,
            'inputs':{
                'input_ids': [torch.tensor(inputIds).flatten() for inputIds in fewInputIds],
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten()
            }
        }

    def PreprocessForQFSTest(self, example, compression, task):
        '''Assuming the max input length is 512'''
        example['task'] = task
        prompt, contexts, _ = self.__promptGather.ConstructCprsedPrompts(example)
        prompt = prompt.strip()

        contextIds = [self.__tokenizer.encode(text=context, add_special_tokens=False) for context in contexts]
        promptIds = self.__tokenizer.encode(text=prompt)
        contextIds = contextIds[0] + contextIds[1] #+ contextIds[2] #+ contextIds[3] + contextIds[4] + contextIds[5] + contextIds[6] + contextIds[7] + contextIds[8] + contextIds[9]
        contextIds = contextIds[ :self.__maxContextLength]
        sepIds = self.__tokenizer.encode('The summary is:\n', add_special_tokens=False)

        ratio = 12
        p = len(contextIds) // 2 + 1
        cprsed_nums = math.ceil(p/ratio)
        cprsed = [4] * cprsed_nums
        holder = [5] * cprsed_nums
    
           
        if compression == 'linear':
            #compre wo prompt
            pass
           
        elif compression == 'partial':
            #compre wo prompt
            # inputIds = [3] + contextIds[p:] + cprsed
            # c_inputIds = promptIds + contextIds[:p] + cprsed + sepIds
            
            #front uncompressed

            inputIds = promptIds + [3] + contextIds[p:] + cprsed
            c_inputIds = promptIds + contextIds[p:] + cprsed + sepIds
            # behind uncompressed
            # inputIds = promptIds + [3] + contextIds[:p] + cprsed
            # c_inputIds = promptIds + contextIds[:p] + cprsed + sepIds
        

        return{
            'op': compression,
            'inputs':{
                'input_ids': torch.tensor(inputIds).flatten(),
            },
            'c_inputs':{
                'input_ids': torch.tensor(c_inputIds).flatten(),
            }
        }


# if __name__ == '__main__':
#     import json
#     from fastchat.model.model_adapter import get_conversation_template
#     from utils import PromptGather
#     from transformers import LlamaTokenizer
#     #/data1/gj/PromptCompression-vicuna/dataset/duc_test_data.json
#     data = json.load(open('/data1/gj/test_sbert_xsum_context.json'))
#     t = LlamaTokenizer.from_pretrained('/data1/gj/vicuna-7b-v1.3')
#     p = promptGather = PromptGather(['Summarize the following text', 'Combining the Dialogue, answer the following question'])
#     x = DatasetModule(data, get_conversation_template('vicuna').system, t,'cat', 'train' ,1024, 128, 1024, 32, 'baseline', p)

#     for i in range(0, len(data)):
#         y = x[i]  
#         #print(y['inputs']['input_ids'].shape, y['inputs']['attention_mask'].shape)
#         #print(y['c_inputs']['input_ids'].shape, y['c_inputs']['attention_mask'].shape)