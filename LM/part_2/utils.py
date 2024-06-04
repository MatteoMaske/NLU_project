# Add functions or classes used for data loading and preprocessing
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# This class computes and stores our vocab
# Word to ids and ids to word
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output



class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    DEVICE = 'cuda'
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

def preprocess_data():

    train_raw = read_file("../dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("../dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("../dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    return train_dataset, dev_dataset, test_dataset, lang

def sum_weights(model, weight_sum):
    for parameter in model.parameters():
        weight_sum[parameter] += parameter.data
    return weight_sum

def save_model(model, exp_name):
    import os

    os.makedirs(os.path.join('bin', exp_name), exist_ok=True)

    path = os.path.join('bin', exp_name, 'best_model.pt')

    torch.save(model.state_dict(), path)
    print("Saving model in", path)

def plot_losses(losses_train, losses_dev, sampled_epochs, exp_name, save=False):

    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o')  # train data
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss', marker='s')  # val data

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()  # Adegua automaticamente i sottografi

    if save: plt.savefig(f"bin/{exp_name}/losses.png")
    plt.show()  # Mostra il grafico

def plot_ppl(ppl_dev_list, sampled_epochs, exp_name, save=False):
    # Assuming ppl_dev_list and sampled_epochs defined
    # some exaples:
    # sampled_epochs = [1, 2, 3, 4, 5]
    # ppl_dev_list = [100, 90, 80, 70, 60]

    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, ppl_dev_list, label='Validation PPL', marker='o')  # val data

    plt.title('Validation PPL')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()  # Adegua automaticamente i sottografi

    if save: plt.savefig(f"bin/{exp_name}/ppl.png")
    plt.show()  # Mostra il grafico

def save_params(args):
    with open(f"bin/{args.exp_name}/params.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(k + ": " + str(v) + "\n")