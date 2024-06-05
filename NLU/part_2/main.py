# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import os
import torch
from tqdm import tqdm
from utils import get_loaders, save_model, plot_stats, save_params, Lang
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='NLU')
    parser.add_argument('--device', type=str, default=device, help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=3, help='Patience')
    parser.add_argument('--runs', type=int, default=1, help='Runs')
    parser.add_argument('--clip', type=float, default=5, help='Clip')
    parser.add_argument('--joint_train', action='store_true', help='Joint training')
    parser.add_argument('--mode', type=str, default="train", help='Mode')

    parser.add_argument('--exp_name', type=str, default='exp2_1', help='Experiment name')

    return parser.parse_args()

def main(args):
    if args.mode == "train":
        train(args)
    else:
        test(args)

def test(args):
    lang = Lang([],[],[])
    model, criterion_slots, criterion_intents, lang = get_checkpoint(args, lang)

    _, _, test_loader, lang = get_loaders(args.batch_size, lang)

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                            criterion_intents, model, lang)
    print("Intent Acc", intent_test['accuracy'])
    print("Slot F1", results_test['total']['f'])


def train(args):

    n_epochs = args.epochs
    runs = args.runs

    slot_f1s, intent_acc = [], []
    for x in range(0, runs):

        train_loader, dev_loader, test_loader, lang = get_loaders()
        model, optimizer, criterion_slots, criterion_intents = get_model(args, lang)

        patience = args.patience
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        
        print("Run", x+1, "="*50)
        for ep in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots,
                            criterion_intents, model)
            if ep % 5 == 0:
                sampled_epochs.append(ep)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                print("Slot F1", f1)
                print("Intent Acc", intent_res['accuracy'])

                if f1 > best_f1:
                    best_f1 = f1
                    save_model(model, optimizer, lang, args.exp_name)
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    save_model(model, optimizer, lang, args.exp_name)
    plot_stats(sampled_epochs, losses_train, losses_dev, args.exp_name)
    save_params(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
