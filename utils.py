import os
import json
import torch
import numpy as np
from fastchat.model.model_adapter import get_conversation_template
class PromptGather:
    def __init__(self, descriptions = []) -> None:
        self.__sumDescription = descriptions[0]
        self.__qaDescription = descriptions[1]
        self.__fixExamples = []
    def ConstructCprsedPrompts(self, example):
        if example['task'] == 'qa':
            # context = '\n'.join(example['Dialogue'])
            # prompt =  '''The above is an entire conversation.\nCombine the above conversation and answer the follow question.\nQuestion: {}:"{}"'''.format(example['Question'][:-1], example['Target'])
            # target =  example['Choices'][example['Human Written Answer'][0]]
            
            context = '\n'.join(example['Dialogue'])
            prompt = '''{}: {}: "{}"\nDialogues: '''.format(self.__qaDescription, example['Question'][:-1], example['Target'])
            target =  example['Choices'][example['Human Written Answer'][0]]
            
        elif example['task'] == 'sum':
            context = example['document']
            prompt = '{}: '.format(self.__sumDescription)
            target = example['summary']
            
            
        elif example['task'] == 'ins':
            context = example['input']
            prompt = example['prompt']
            target = example['response']
            
        elif example['task'] == 'arxiv':
            context = example['article']
            prompt = '{}: '.format(self.__sumDescription)
            target = example['abstract']
            
        elif example['task'] == 'context-sum':
            contexts = []
            targets = []
            for i in range(example['fewshot']):
                similarText = example[f'similarText{i}']
                similarSummary = example['similarSummary0'] if 'similarSummary0' in example else example['similarTarget0']
                target = example['target']
                # context = 'USER: {}: {} ASSISTANT: {}'.format(self.__sumDescription, similarText, similarSummary)
                context = '{}: {} -> {}'.format(self.__sumDescription, similarText, similarSummary)
                contexts.append(context)
                targets.append(target)
            context = contexts
            prompt = '{}: {}'.format(self.__sumDescription, example['source'])
            
            
        elif example['task'] == 'context-fix':
            contexts = []
            targets = []
            for i in range(example['fewshot']):
                similarText = example[f'similarText{i}'] 
                similarSummary = example['similarSummary0'] if 'similarSummary0' in example else example['similarTarget0']
                #context = 'USER: {}: {} ASSISTANT: {}'.format(self.__sumDescription, similarText, similarSummary)
                context = '{}: {} -> {}'.format(self.__sumDescription, similarText, similarSummary)

                target = example['target']
                contexts.append(context)
                targets.append(target)
            
            if self.__fixExamples == []:
                self.__fixExamples = contexts
            else:
                contexts = self.__fixExamples
            context = contexts
            prompt = '{}: {}'.format(self.__sumDescription, example['source'])

        elif example['task'] == 'context-qa':
            contexts = []
            targets = []
            for i in range(example['fewshot']):
                similarQuery = example[f'similarQuery{i}']
                similarText = example[f'similarText{i}']
                similarTarget = example[f'similarTarget{i}']
                target = example['target']
                context = 'USER: Combining the document, answer of the following question with explanation: {}\nDocument: {} ASSISTANT: {}'.format(similarQuery, similarText, similarTarget)
                #context = 'USER: Combining the document, answer the following question: {}\nDocument: {} ASSISTANT: {}'.format(similarQuery, similarText, similarTarget)
                #context = 'USER: According to the document: {} Answer this question: {} ASSISTANT: {}'.format(similarText, similarQuery, similarTarget)
                contexts.append(context)
                targets.append(target)
            context = contexts
            prompt = 'Combining the document, answer the following question with explanation: {}\nDocument: {}'.format(example['query'], example['source'])
            #prompt = 'Combining the document, answer the following question: {}\nDocument: {}'.format(example['source'], example['query'])
            #prompt = 'According to the document: {} Answer this question: {}'.format(example['source'], example['query'])
        
        elif example['task'] == 'context-cot':
            contexts = []
            targets  = []
            for i in range(example['fewshot']):
                s = example[f'similar{i}']
                similarQuery = s['passage'] + s['question']
                similarSolution = s['other']['solution']
                contexts.append('USER: {} Options: {} ASSISTANT: {}'.format(similarQuery, s['options'] ,similarSolution))
            context = contexts
            prompt = '{} {} Options: {}'.format(example['passage'], example['question'], example['options']).strip()
            target = example['label']
            
        elif example['task'] == 'qfs':
            topic = example['topic'].strip()
            prompt = f'According the topic: {topic} Summarize the documents:'
            context = example['document']
            target = example['summary']
        elif example['task'] == 'legal':
            prompt = "请根据以下案情写判决书:"
            context = example['input']
            target = example['output']
        else:
            raise NotImplementedError
            
        return prompt, context, target

    def ConstructBaselinePrompts(self, example):
        if 'qa' == example['task']:
            #context = '\n'.join(example['Dialogue'])
            # prompt = '''{}: {}: "{}"\nDialogues: '''.format(self.__qaDescription, example['Question'][:-1], example['Target'])
            # prompt = ''
            context = '''Here is a conversation, {}\nAccording to this target conversation: "{}", answer the follow question: {}. The answer is: '''.format('\n'.join(example['Dialogue']), example['Target'],example['Question'])
        
        elif 'sum' == example['task']:
            context = example['document']
            prompt = '{}: '.format(self.__sumDescription)
        elif example['task'] == 'ins':
            context = example['input']
            prompt = example['prompt']
        elif example['task'] == 'arxiv':
            context = example['article']
            prompt = '{}: '.format(self.__sumDescription)
        elif example['task'] == 'context-sum':
            similarText = example['similarText0']
            similarSummary = example['similarSummary0'] if 'similarSummary0' in example else example['similarTarget0']
            prompt = '{}: {} ->'.format(self.__sumDescription, example['source'])
            #context = 'USER: {}: {} ASSISTANT: {}'.format(self.__sumDescription, similarText, similarSummary)
            context = '{}: {} -> {}'.format(self.__sumDescription, similarText, similarSummary)
        
        elif example['task'] == 'context-fix':
            contexts = []
            targets = []
            if compressed_prompt is None:
                for i in range(example['fewshot']):
                    similarText = example[f'similarText{i}'] 
                    similarSummary = example['similarSummary0'] if 'similarSummary0' in example else example['similarTarget0']
                    
                    #context = 'USER: {}: {} ASSISTANT: {}'.format(self.__sumDescription, similarText, similarSummary)
                    #context = '{}: {} -> {}'.format(self.__sumDescription, similarText, similarSummary)
                    target = example['target']
                    contexts.append(context)
                    targets.append(target)
            
                if self.__fixExamples == []:
                    self.__fixExamples = contexts
                else:
                    contexts = self.__fixExamples
            context = contexts if compressed_prompt is None else compressed_prompt
            prompt = '{}: {}'.format(self.__sumDescription, example['source'])
            
        elif example['task'] == 'context-qa':
            similarQuery = example['similarQuery0']
            similarText = example['similarText0']
            similarTarget = example['similarTarget0']
            
            prompt = 'Combining the document, answer the following question: {}\nDocument: {}'.format(example['query'], example['source'])
            context = 'USER: Combining the document, answer the following question: {}\nDocument: {} ASSISTANT: {}'.format(similarQuery, similarText, similarTarget)

            # prompt = 'Combining the document, answer the following question: {}\nDocument: {}'.format(example['source'], example['query'])
            # context = 'USER: Combining the document, answer the following question: {}\nDocument: {} ASSISTANT: {}'.format(similarQuery, similarText, similarTarget)

            #prompt = 'According to the document: {} Answer this question: {}'.format(example['source'], example['query'])
            #context = 'USER: According to the document: {} Answer this question: {} ASSISTANT: {}'.format(similarText, similarQuery ,similarTarget)
        
        elif example['task'] == 'qfs':
            topic = example['topic'].strip()
            document = example['document'][0] + example['document'][1]
            prompt = f'According the topic: {topic} Summarize the documents:'
            context = document
            
        elif example['task'] == 'context-cot':
            context = ''
            for i in range(example['fewshot']):
                s = example[f'similar{i}']
                similarQuery = s['passage'] + s['question']
                similarSolution = s['other']['solution']
                context += 'USER: {} Options: {} ASSISTANT: {}'.format(similarQuery, s['options'] ,similarSolution)
            
            prompt = '{} {} Options: {}'.format(example['passage'], example['question'], example['options']).strip()
        elif example['task'] == 'legal':
            prompt = '请根据以下案情写判决书:'
            context = example['input']
        
        else:
            raise NotImplementedError
            
        return prompt, context
def GlobalSeed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def LoadDatas(filePath) -> None:
    return json.load(open(filePath, 'r', encoding='utf8'))
