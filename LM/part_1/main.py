# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import collate_fn, preprocess_data, save_model

from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import argparse
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--hid_size", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--clip", type=int, default=5)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--emb_dropout", type=float, default=0.5)
    parser.add_argument("--out_dropout", type=float, default=0.2)
    parser.add_argument("--weight_tying", type=bool, default=False)
    parser.add_argument("--dropout_type", type=str, default='variational')
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--exp_name", type=str, default='exp1_2')
    parser.add_argument("--mode", type=str, default='test')

    return parser.parse_args()

def main(args):
    if args.mode == 'train':
        train(args)
    else:
        test(args)

def test(args):
    _,_, test_dataset, lang = preprocess_data()
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    model, criterion_eval = get_checkpoint(args, lang)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)
    print('Test ppl: ', final_ppl)


def train(args):

    # Parse the arguments
    
    clip = args.clip
    device = 'cuda'

    # Load the data
    train_dataset, dev_dataset, test_dataset, lang = preprocess_data()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Load the model, the optimizer, and the criterion
    model, optimizer, criterion_train, criterion_eval = create_model(args.emb_size, args.hid_size, args.lr, args.clip, args.device, args.emb_dropout, args.out_dropout, args.weight_tying, args.dropout_type, lang, args.optimizer)

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    ppl_dev_list = []
    best_ppl = math.inf
    best_model = None
    best_val_loss = []
    stored_loss = 100000000
    pbar = tqdm(range(1,args.n_epochs))

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())

            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = args.patience
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    save_model(best_model, args.exp_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)