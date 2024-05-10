import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertModel

class Bert(BertPreTrainedModel):
    def __init__(self, config, out_int, out_slot, dropout):
        super(Bert, self).__init__(config)

        self.bert = BertModel(config=config)

        self.slot_out = nn.Linear(config.hidden_size, out_slot)
        self.intent_out = nn.Linear(config.hidden_size, out_int)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Dropout(0.1)

    def forward(self, utt_emb, att_mask):

        output = self.bert(utt_emb, attention_mask=att_mask)

        sequence_output = output.last_hidden_state
        last_hidden = output.last_hidden_state[:,0,:] # [CLS]

        # print("utt_enc", sequence_output.shape)
        # print("last_hidden", last_hidden.shape)

        slots = self.slot_out(sequence_output)
        intent = self.intent_out(last_hidden)

        slots = slots.permute(0,2,1)
        # print("slot_out", slots.shape)
        # print("intent_out", intent.shape)

        return slots, intent
