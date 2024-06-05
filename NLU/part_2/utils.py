# Add functions or classes used for data loading and preprocessing
import json, os
from pprint import pprint
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import BertTokenizer

PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def create_split(tmp_train_raw, test_raw):

    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    # # Intent distributions
    # print('Train:')
    # pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    # print('Dev:'),
    # pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    # print('Test:')
    # pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    # print('='*89)
    # # Dataset size
    # print('TRAIN size:', len(train_raw))
    # print('DEV size:', len(dev_raw))
    # print('TEST size:', len(test_raw))
    return train_raw, dev_raw

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=0, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab


import torch
import torch.utils.data as data

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        #Assignment 2 - BERT
        self.tokenizer = tokenizer
        self.CLS_TOKEN = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.SEP_TOKEN = self.tokenizer.convert_tokens_to_ids('[SEP]')

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids, self.slot_ids = self.mapping_seq_tokenizer(self.utterances, self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        att_mask = torch.ones(utt.shape)
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'att_mask': att_mask, 'slots': slots, 'intent': intent}
        return sample

    # Auxiliary methods
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq_tokenizer(self, utterances, slots, slot_mapper): # Map sequences to number
        utt_tokenized, slots_tokenized = [], []
        discarded = 0
        for seq, seq_slots in zip(utterances, slots):
            seq_tokens = []
            slot_tokens = []
            # print(seq, seq_slots)
            for word, slot in zip(seq.split(), seq_slots.split()):
                word_tokens = self.tokenizer(word)['input_ids']
                seq_tokens += word_tokens[1:-1]
                slot_tokens.append(slot_mapper[slot])

                padding = len(word_tokens)-3
                if padding:
                    slot_tokens.extend([PAD_TOKEN]*padding)
            # print(tokenizer.convert_ids_to_tokens(seq_tokens), [lang.id2slot[slot] for slot in slot_tokens])

            assert len(slot_tokens) == len(seq_tokens), f"{self.tokenizer.convert_ids_to_tokens(seq_tokens)}{len(seq_tokens)} --- {[tok for tok in slot_tokens]}{len(slot_tokens)}"

            seq_tokens = [self.CLS_TOKEN] + seq_tokens + [self.SEP_TOKEN]
            slot_tokens = [PAD_TOKEN] + slot_tokens + [PAD_TOKEN]

            if len(slot_tokens) == len(seq_tokens):
                utt_tokenized.append(seq_tokens)
                slots_tokenized.append(slot_tokens)
            else:
                print(seq_tokens)
                print(slot_tokens)
                print("------------")
                discarded+=1
        print(discarded)
        return utt_tokenized, slots_tokenized

    
def prepare_dataset(lang=None):

    abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    dataset_dir = os.path.join(abs_path, 'dataset')

    tmp_train_raw = load_data(os.path.join(dataset_dir,'ATIS','train.json'))
    test_raw = load_data(os.path.join(dataset_dir,'ATIS','test.json'))
    # print('Train samples:', len(tmp_train_raw))
    # print('Test samples:', len(test_raw))

    # pprint(tmp_train_raw[0])
    train_raw, dev_raw = create_split(tmp_train_raw, test_raw)

    if not lang:
        words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                                    # the cutoff
        corpus = train_raw + dev_raw + test_raw # We do not want unk labels,
                                                # however this depends on the research purpose
        slots = set(sum([line['slots'].split() for line in corpus],[]))
        intents = set([line['intent'] for line in corpus])

        lang = Lang(words, intents, slots, cutoff=0)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    return train_dataset, dev_dataset, test_dataset, lang

def get_loaders(batch_size=128, lang=None):
    train_dataset, dev_dataset, test_dataset, lang = prepare_dataset(lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, lang

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    att_mask, _ = merge(new_item['att_mask'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    att_mask = att_mask.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt
    new_item["att_mask"] = att_mask
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

def save_model(model, optimizer, lang, exp_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir,'bin', exp_name), exist_ok=True)

    path = os.path.join(current_dir, 'bin', exp_name, 'checkpoint')

    saving_object = {"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "w2id": "BertTokenizer",
                    "slot2id": lang.slot2id,
                    "intent2id": lang.intent2id}
    torch.save(saving_object, path)
    print("Saving model in", path)

def plot_stats(sampled_epochs, losses_train, losses_dev, exp_name):
    import matplotlib.pyplot as plt

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path=os.path.join(current_dir,'bin', exp_name)
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')

    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()

    plt.savefig(os.path.join(path, 'losses.png'))

    plt.show()

def save_params(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'bin', args.exp_name, 'params.txt')
    with open(path, 'w') as f:
        for k, v in vars(args).items():
            f.write(k + ": " + str(v) + "\n")
    print("Saving parameters in", path)