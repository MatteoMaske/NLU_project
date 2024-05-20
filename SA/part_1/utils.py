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
    """
    input: path/to/data
    output: json
    """
    dataset = []
    with open(path, encoding="utf-8") as f:
        dataset = f.readlines()
    return dataset


def preprocess_data(data):
    """
    input: json
    output: list of dict
    """
    new_data = []
    for line in data:
        utt, aspects = line.split("####")
        new_utt, new_aspects = [], []
        for aspect in aspects.split():
            word, aspect = aspect.rsplit("=", 1)
            new_utt.append(word)
            new_aspects.append(aspect)
        assert len(new_utt) == len(
            new_aspects
        ), f"Utterance and slots have different lengths {utt} != {new_aspects}"
        new_data.append({"utterance": utt, "aspects": " ".join(new_aspects)})
    return new_data


def create_split(tmp_train_raw, test_raw):

    portion = 0.10
    # Random Stratify
    train_raw, dev_raw = train_test_split(
        tmp_train_raw, test_size=portion, random_state=42, shuffle=True
    )

    # Dataset size
    print("TRAIN size:", len(train_raw))
    print("DEV size:", len(dev_raw))
    print("TEST size:", len(test_raw))
    return train_raw, dev_raw


class Lang:
    def __init__(self, words, aspects, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.aspect2id = self.lab2id(aspects)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2aspect = {v: k for k, v in self.aspect2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {"pad": PAD_TOKEN}
        if unk:
            vocab["unk"] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class Aspects (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.aspects = []
        self.unk = unk

        #Assignment 2 - BERT
        self.tokenizer = tokenizer
        self.CLS_TOKEN = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.SEP_TOKEN = self.tokenizer.convert_tokens_to_ids('[SEP]')

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.aspects.append(x['aspects'])

        self.utt_ids, self.aspects_ids = self.mapping_seq_tokenizer(self.utterances, self.aspects, lang.aspect2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        att_mask = torch.ones(utt.shape)
        aspects = torch.Tensor(self.aspects_ids[idx])
        sample = {'utterance': utt, 'att_mask': att_mask, 'aspects': aspects}
        return sample

    def mapping_seq_tokenizer(self, utterances, aspects, aspect_mapper): # Map sequences to number
        utt_tokenized, aspects_tokenized = [], []
        discarded = 0
        for seq, seq_aspects in zip(utterances, aspects):
            seq_tokens = []
            aspect_tokens = []
            for word, aspect in zip(seq.split(), seq_aspects.split()):
                word_tokens = self.tokenizer(word)['input_ids']
                seq_tokens += word_tokens[1:-1]
                aspect_tokens.append(aspect_mapper[aspect])

                padding = len(word_tokens)-3
                if padding:
                    aspect_tokens.extend([PAD_TOKEN]*padding)
            # print(tokenizer.convert_ids_to_tokens(seq_tokens), [lang.id2slot[slot] for slot in slot_tokens])

            assert len(aspect_tokens) == len(seq_tokens), f"{self.tokenizer.convert_ids_to_tokens(seq_tokens)}{len(seq_tokens)} --- {[tok for tok in aspect_tokens]}{len(aspect_tokens)}"

            seq_tokens = [self.CLS_TOKEN] + seq_tokens + [self.SEP_TOKEN]
            aspect_tokens = [PAD_TOKEN] + aspect_tokens + [PAD_TOKEN]

            if len(aspect_tokens) == len(seq_tokens):
                utt_tokenized.append(seq_tokens)
                aspects_tokenized.append(aspect_tokens)
            else:
                print(seq_tokens)
                print(aspect_tokens)
                print("------------")
                discarded+=1
        if discarded: print(discarded)
        return utt_tokenized, aspects_tokenized


def prepare_dataset():
    abs_path = os.path.dirname(
        os.path.abspath(os.path.dirname(__file__))
    )  # SA directory
    tmp_train_raw = load_data(os.path.join(abs_path, "dataset", "laptop14_train.txt"))
    test_raw = load_data(os.path.join(abs_path, "dataset", "laptop14_test.txt"))
    # print('Train samples:', len(tmp_train_raw))
    # print('Test samples:', len(test_raw))

    tmp_train_raw = preprocess_data(tmp_train_raw)
    test_raw = preprocess_data(test_raw)

    # pprint(tmp_train_raw[0])
    train_raw, dev_raw = create_split(tmp_train_raw, test_raw)

    words = sum(
        [x["utterance"].split() for x in train_raw], []
    )  # No set() since we want to compute
    # the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not want unk labels,
    # however this depends on the research purpose
    aspects = set(sum([line["aspects"].split() for line in corpus], []))
    # intents = set([line['intent'] for line in corpus])

    lang = Lang(words, aspects, cutoff=0)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create our datasets
    train_dataset = Aspects(train_raw, lang, tokenizer)
    dev_dataset = Aspects(dev_raw, lang, tokenizer)
    test_dataset = Aspects(test_raw, lang, tokenizer)

    return train_dataset, dev_dataset, test_dataset, lang


def get_loaders(batch_size=128):
    train_dataset, dev_dataset, test_dataset, lang = prepare_dataset()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, lang


def collate_fn(data):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = (
            padded_seqs.detach()
        )  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["utterance"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item["utterance"])
    att_mask, _ = merge(new_item['att_mask'])
    y_aspects, y_lengths = merge(new_item["aspects"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_utt = src_utt.to(device)  # We load the Tensor on our selected device
    att_mask = att_mask.to(device)
    y_aspects = y_aspects.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt
    new_item["att_mask"] = att_mask
    new_item["y_aspects"] = y_aspects
    new_item["aspects_len"] = y_lengths
    return new_item

def save_model(model, optimizer, lang, exp_name):
    os.makedirs(os.path.join('NLU/results', exp_name), exist_ok=True)

    path = os.path.join('NLU/results', exp_name, 'checkpoint')

    saving_object = {"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "w2id": "BertTokenizer",
                    "slot2id": lang.slot2id,
                    "intent2id": lang.intent2id}
    torch.save(saving_object, path)
    print("Saving model in", path)

def plot_stats(losses_train, losses_dev, sampled_epochs, exp_name):
    import matplotlib.pyplot as plt
    path=os.path.join('NLU/results', exp_name)
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')

    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()

    plt.savefig(os.path.join(path, 'losses.png'))

    plt.show()