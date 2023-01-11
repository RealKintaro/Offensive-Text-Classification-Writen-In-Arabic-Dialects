
from transformers import AutoModel
from torch import nn
import pytorch_lightning as pl


class MediumBert(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained('asafaya/bert-medium-arabic')
        self.fc = nn.Linear(512,18)
    
    def forward(self,input_ids,attention_mask):
        out = self.bert_model(input_ids = input_ids, attention_mask =attention_mask)#inputs["input_ids"],inputs["token_type_ids"],inputs["attention_mask"])
        pooler = out[1]
        out = self.fc(pooler)
        return out