# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import collate_fn, preprocess_data, sum_weights, save_model, plot_losses, plot_ppl, save_params

from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import argparse
import numpy as np
import math


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_size", type=int, default=600)
    parser.add_argument("--hid_size", type=int, default=600)
    parser.add_argument("--lr", type=float, default=10)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--lr_decay_epoch", type=int, default=3)
    parser.add_argument("--clip", type=int, default=5)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--emb_dropout", type=float, default=0.6)
    parser.add_argument("--out_dropout", type=float, default=0.6)
    parser.add_argument("--weight_tying", type=bool, default=True)
    parser.add_argument("--dropout_type", type=str, default='variational')
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--NT_ASGD", type=bool, default=False)

    return parser.parse_args()


def main():

    # Parse the arguments
    args = parse_args()
    clip = args.clip
    device = 'cuda'

    # Load the data
    train_dataset, dev_dataset, test_dataset, lang = preprocess_data()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Load the model, the optimizer, and the criterion
    model, optimizer, scheduler, criterion_train, criterion_eval = create_model(args, device, lang)

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    ppl_dev_list = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,args.n_epochs))
    lr_decay = args.lr_decay
    lr_decay_epoch = args.lr_decay_epoch

    #for NT_ASGD
    best_dev_loss = []
    non_monotonic_trigger = 3
    #for manual ASGD
    compute_avg = False
    iter = 1
    best_weights = {}
    weight_sum = {}
    weight_avg = {}

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss).mean())

        if args.NT_ASGD:
            if compute_avg:
                weight_sum = sum_weights(model, weight_sum)
                iter += 1

                tmp = {}
                for parameter in model.parameters():
                    tmp[parameter] = parameter.data.clone()
                    weight_avg[parameter] = weight_sum[parameter] / iter
                    parameter.data = weight_avg[parameter].clone()
                    print(parameter.data)

                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                ppl_dev_list.append(ppl_dev)
                losses_dev.append(np.asarray(loss_dev).mean())

                for parameter in model.parameters():
                    parameter.data = tmp[parameter].clone()

            else:

                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                ppl_dev_list.append(ppl_dev)
                losses_dev.append(np.asarray(loss_dev).mean())

                if len(best_dev_loss)>non_monotonic_trigger and loss_dev > min(best_dev_loss[:-non_monotonic_trigger]):
                    print("\nSwitching optimizer to ASGD")
                    compute_avg = True
                    lr_decay_epoch = 3
                    lr_decay = 0.5
                    optimizer.param_groups[0]['lr'] = 1

                    #initialize weights for averaging from next iteration
                    weight_sum = best_weights

                best_dev_loss.append(loss_dev)
        else:
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())

        # if (epoch+1) % lr_decay_epoch == 0:
        #     optimizer.param_groups[0]['lr'] *= lr_decay
        #     print(optimizer.param_groups[0]['lr'])
        scheduler.step()

        pbar.set_description("PPL: %f" % ppl_dev)
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            for param in model.parameters():
                best_weights[param] = param.data.clone()
            patience = args.patience
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    # Save the model
    save_model(best_model)
    plot_losses(losses_train, losses_dev, sampled_epochs, save=True)
    plot_ppl(ppl_dev_list, sampled_epochs, save=True)
    save_params(args)


if __name__ == "__main__":
    main()