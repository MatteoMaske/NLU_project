import torch
import torch.nn as nn
from model import Bert
from utils import PAD_TOKEN
import torch.optim as optim
from transformers import BertConfig, BertTokenizer

def train_loop(data, optimizer, criterion_aspects, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        aspects = model(sample['utterances'], sample['att_mask'])
        loss_aspect = criterion_aspects(aspects, sample['y_aspects'])

        loss_array.append(loss_aspect.item())
        loss_aspect.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_aspects, model, lang):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
    loss_array = []

    ref_aspects = []
    hyp_aspects = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            aspects = model(sample['utterances'], sample['att_mask'])
            loss_aspect = criterion_aspects(aspects, sample['y_aspects'])
            loss_array.append(loss_aspect.item())

            # Slot inference
            output_aspects = torch.argmax(aspects, dim=1)
            for id_seq, seq in enumerate(output_aspects):
                length = sample['aspects_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_aspects'][id_seq][:length].tolist()
                gt_aspects = [lang.id2aspect[elem] for elem in gt_ids]
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                to_decode = seq[:length].tolist()

                new_utterance = []
                new_gt_aspects = []
                new_to_decode = []

                #removing padding token from the gt_slots, ref_slots and utterance
                for index, aspect in enumerate(gt_aspects):
                    if aspect != 'pad':
                        new_gt_aspects.append(aspect)
                        new_utterance.append(utterance[index])
                        new_to_decode.append(to_decode[index])

                gt_aspects = new_gt_aspects
                utterance = new_utterance
                to_decode = new_to_decode

                ref_aspects.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_aspects)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2aspect[elem]))
                hyp_aspects.append(tmp_seq)
    try:
        results = evaluate(ref_aspects, hyp_aspects)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_aspects])
        hyp_s = set([x[1] for x in hyp_aspects])
        print("Found those slots not in the ground truth", hyp_s.difference(ref_s))
        print(f"{len(hyp_s.difference(ref_s))} out of {len(hyp_s)}")
        results = {"total":{"f":0}}
    return results, loss_array

def evaluate(ref_aspects, pred_aspects):
    assert len(ref_aspects) == len(pred_aspects)
    n_samples = len(ref_aspects)
    # number of true positive, gold standard, predicted opinion targets
    n_tp_aspects, n_gt_aspects, n_pred_aspects = 0, 0, 0
    for i in range(n_samples):
        gt_aspects = ref_aspects[i]
        p_aspects = pred_aspects[i]
        # hit number
        n_hit = 0
        for t in p_aspects:
            if t[1] != 'O' and t in gt_aspects:
                n_hit += 1

        n_tp_aspects += n_hit
        n_gt_aspects += sum([1 for a in gt_aspects if a[1] != 'O'])
        n_pred_aspects += sum([1 for a in p_aspects if a[1] != 'O'])
    # print(ref_aspects[:5])
    # print(pred_aspects[:5])
    # add 0.001 for smoothing
    # calculate precision, recall and f1 for ote task
    precision = float(n_tp_aspects) / float(n_pred_aspects + 1e-3)
    recall = float(n_tp_aspects) / float(n_gt_aspects + 1e-3)
    f1 = 2 * precision * recall / (precision + recall + 1e-3)
    scores = (precision, recall, f1)
    return scores

def init_weights(mat):
    for n,m in mat.named_modules():
        if type(m) in [nn.Linear]:
            if "aspect" in n:
                print("Initializing", n)
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def get_model(args, lang):

    lr = args.lr

    out_aspect = len(lang.aspect2id)

    dropout=args.dropout

    config = BertConfig(hidden_dropout_prob=dropout)
    model = Bert.from_pretrained("bert-base-uncased", config=config, out_aspect=out_aspect, dropout=dropout).to(args.device)
    model.apply(init_weights)
    # print(model)

    if args.joint_training:
        param_group = model.parameters()
        print("Joint training")
    else:
        param_group = [p for n,p in model.named_parameters() if "aspect" in n]
        
    optimizer = optim.Adam(param_group, lr=lr)

    criterion_aspects = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    return model, optimizer, criterion_aspects

def get_checkpoint(args, lang):
    import os

    checkpoint_dir = os.path.join("bin", args.exp_name)

    if not os.path.exists(checkpoint_dir):
        raise ValueError("Checkpoint not found")
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))

    if args.exp_name == "exp3_1":
        args.joint_training = True

    model, optimizer, criterion_aspects = get_model(args, lang)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, criterion_aspects