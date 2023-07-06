import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModel
from torch.nn import KLDivLoss


class ModelModule(nn.Module):
    def __init__(self, modelNameOrPath) -> None:
        super(ModelModule, self).__init__()
        self.__model = AutoModel.from_pretrained(modelNameOrPath, trust_remote_code=True, revision='').half().transformer
        self.__embedding = self.__model.get_input_embeddings()
        self.__lossFn = KLDivLoss(reduction='batchmean', log_target=False)
        self.__mlp = nn.Linear(4096, 4096)
        
    def forward(self, input_ids, c_input_ids, labels, c_labels):
        input_embeds = self.__embedding(input_ids)
        c_input_embeds = self.__embedding(c_input_ids)
        
        attention_mask = self.__model.get_masks(input_ids, input_ids.device)
        c_attention_mask = self.__model.get_masks(c_input_ids, input_ids.device)
        
        MASK, gMASK = self.__model.config.mask_token_id, self.__model.config.gmask_token_id
        seqs = input_ids.tolist()

        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        position_ids = self.__model.get_position_ids(
            input_ids,
            mask_positions=mask_positions,
            device=input_ids.device,
            use_gmasks=use_gmasks
        )
        
        c_seqs = c_input_ids.tolist()

        c_mask_positions, c_use_gmasks = [], []
        for seq in c_seqs:
            c_mask_token = gMASK if gMASK in seq else MASK
            c_use_gmask = c_mask_token == gMASK
            c_mask_positions.append(seq.index(mask_token))
            c_use_gmasks.append(c_use_gmask)

        c_position_ids = self.__model.get_position_ids(
            c_input_ids,
            mask_positions=c_mask_positions,
            device=c_input_ids.device,
            use_gmasks=c_use_gmasks
        )
        with torch.no_grad():
            oriHiddenStates = self.__model(inputs_embeds = input_embeds,
                                        attention_mask = attention_mask,
                                        position_ids = position_ids)[0].permute(1, 0, 2)
        
        
        mask_shape = list(oriHiddenStates.shape)
        mask_shape[1] = -1
        maskHiddenStates = oriHiddenStates[input_ids == self.__model.config.mask_token_id, ...].view(*mask_shape)
        
        maskHiddenStates = self.__mlp(maskHiddenStates.float())
        
        labelHiddenStates = oriHiddenStates[labels != -100, ...].view(-1, oriHiddenStates.shape[-1])
        target = F.softmax(labelHiddenStates, dim=-1)
        
        c_input_embeds[c_input_ids == self.__model.config.mask_token_id, ...] = maskHiddenStates.half().view(-1, c_input_embeds.shape[-1])
        
        #with torch.no_grad():
        cprsedHiddenStates = self.__model(inputs_embeds = c_input_embeds,
                                    attention_mask = c_attention_mask,
                                    position_ids = c_position_ids)[0].permute(1, 0, 2)
        outHiddenStates = cprsedHiddenStates[c_labels != -100, ...].view(-1, cprsedHiddenStates.shape[-1])
        preds = F.log_softmax(outHiddenStates, dim=-1)
        
        loss = self.__lossFn(preds, target)
        #compressedHiddenstate
        
        return loss

    def FreezeBackbone(self):
        for parm in self.__model.parameters():
                parm.requires_grad = False
            
            
    def SaveTrainedModel(self, val: float ,path: str = './') -> None:
        os.makedirs(path, exist_ok = True)
        torch.save(self, os.path.join(path, 'model_f1_{0:.4f}.pth'.format(val)))
    
    def ConfigOptimizer(self, lr: float) -> Adam:
        return Adam(self.parameters(), lr=lr)
    