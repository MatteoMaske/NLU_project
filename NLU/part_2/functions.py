import torch
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report
from model import Bert
from utils import PAD_TOKEN
import torch.optim as optim
from transformers import BertConfig, BertTokenizer

from conll import evaluate
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['att_mask'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        # assert slots.shape == sample['y_slots'].shape, f"slots have different shapes: {slots.shape}, {sample['y_slots'].shape}"
        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['att_mask'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                to_decode = seq[:length].tolist()

                #removing padding token from the gt_slots, ref_slots and utterance
                for index, slot in enumerate(gt_slots):
                    if slot == 0: # <PAD> token
                        utterance.pop(index)
                        to_decode.pop(index)
                gt_slots = [x for x in gt_slots if x != 0]

                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print("Found those slots not in the ground truth", hyp_s.difference(ref_s))
        print(f"{len(hyp_s.difference(ref_s))} out of {len(hyp_s)}")
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def init_weights(mat):
    for n,m in mat.named_modules():
        if type(m) in [nn.Linear]:
            if "slot" in n or "intent" in n:
                print("Initializing", n)
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def get_model(args, lang):

    lr = args.lr

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    dropout=args.dropout

    config = BertConfig(hidden_dropout_prob=dropout)
    model = Bert.from_pretrained("bert-base-uncased", config=config, out_int=out_int, out_slot=out_slot, dropout=dropout).to(device)
    model.apply(init_weights)
    print(model)

    param_group = {'params': [p for n,p in model.named_parameters() if ("slot" in n or "intent" in n)]}
    optimizer = optim.Adam(param_group, lr=lr)

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    return model, optimizer, criterion_slots, criterion_intents