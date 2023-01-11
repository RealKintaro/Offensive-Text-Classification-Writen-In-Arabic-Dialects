import torch.nn as nn
from transformers import BertModel
import pytorch_lightning as pl

BERT_MODEL_NAME = 'alger-ia/dziribert'
class Dialect_Detection(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask)
        output = self.classifier(output.pooler_output)                    
        # if provided with labels return loss and output
        if labels is not None:
            loss = self.criterion(output, labels)
            return loss, output 

        return output