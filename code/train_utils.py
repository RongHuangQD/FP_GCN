import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn import metrics as met

def train(model, device, loader, optimizer):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """
    curve = list()
    model.train()
    num_skips = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        num_samples = batch.x.size(0)
        if num_samples <= 1:
            num_skips += 1
            if float(num_skips) / len(loader) >= 0.25:
                logging.warning("Warning! 25% of the batches were skipped this epoch")
            continue

        if num_samples < 10:
            logging.warning("Warning! BatchNorm applied on a batch "
                            "with only {} samples".format(num_samples))
        optimizer.zero_grad()
        pred = model(batch)
        targets = batch.y.view(-1, )
        mask = ~torch.isnan(targets)
        loss_train = torch.nn.CrossEntropyLoss()(pred[mask], targets[mask])#CrossEntropyLoss期望输入的是未归一化的分数（logits），并在内部应用了 log_softmax 来计算损失。因此在模型中没有用到softmax
        loss_train.backward()
        optimizer.step()
        curve.append(loss_train.detach().cpu().item())
    return curve

def eval(model, device, loader):

    model.eval()
    y_true = []
    y_pred = []
    losses = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            targets = batch.y.view(-1, )
            y_true.append(batch.y.detach().cpu())
            mask = ~torch.isnan(targets)
            loss = torch.nn.CrossEntropyLoss()(pred[mask], targets[mask])
            losses.append(loss.detach().cpu().item())
        y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy() if len(y_true) > 0 else None
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {'y_pred': y_pred, 'y_true': y_true}
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
    return accuracy(input_dict), mean_loss, y_true, np.argmax(y_pred, axis=1)

def accuracy(input_dict):
    y_true = input_dict['y_true']
    y_pred = np.argmax(input_dict['y_pred'], axis=1)
    assert y_true is not None
    assert y_pred is not None
    metric = met.accuracy_score(y_true, y_pred)
    return metric
