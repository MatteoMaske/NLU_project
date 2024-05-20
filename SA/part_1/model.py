import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class Bert(BertPreTrainedModel):
    def __init__(self, config, out_aspect, dropout):
        super(Bert, self).__init__(config)

        self.bert = BertModel(config=config)

        self.aspect_out = nn.Linear(config.hidden_size, out_aspect)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Dropout(0.1)

    def forward(self, utt_emb, att_mask):

        output = self.bert(utt_emb, attention_mask=att_mask)

        sequence_output = output.last_hidden_state

        # print("utt_enc", sequence_output.shape)
        # print("last_hidden", last_hidden.shape)

        aspects = self.aspect_out(sequence_output)

        aspects = aspects.permute(0,2,1)
        # print("slot_out", slots.shape)
        # print("intent_out", intent.shape)

        return aspects