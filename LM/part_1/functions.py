# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math
import torch
import torch.nn as nn
import torch.optim as optim
from model import LM_LSTM, LM_RNN

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def create_model(emb_size, hid_size, lr, clip, device, emb_dropout, out_dropout, weight_tying, dropout_type, lang, optimizer_type):

    vocab_len = len(lang.word2id)
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],out_dropout=out_dropout, emb_dropout=emb_dropout).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr) if optimizer_type == 'sgd' else optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    print(model)

    return model, optimizer, criterion_train, criterion_eval

def get_checkpoint(args, lang):

    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir,"bin", args.exp_name)
    
    vocab_len = len(lang.word2id)

    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pt"), map_location=args.device)
    if args.exp_name == "exp1_0":
        model = LM_RNN(300, 300, vocab_len, pad_index=lang.word2id["<pad>"]).to(args.device)
    else:
        model = LM_LSTM(300, 300, vocab_len, pad_index=lang.word2id["<pad>"]).to(args.device)
    model.load_state_dict(checkpoint)

    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    return model, criterion_eval