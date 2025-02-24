from .feature_selection_network import FeatureSelectionNetwork, gumbel_sigmoid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from typing import Literal

def select_gumbel_features(
    X_train,
    y_train,
    device="cpu",
    verbose=False,
    temperature_decay=0.997,
    epochs=300,
    batch_size=1,
    fs_balance=1.0,
    target_features_mode: Literal["auto", "target", "raw"] = "auto",
    mode: Literal["classification", "regression"] = "classification",
):
    if mode == "classification":
        balance = (y_train.sum(axis=0) / len(y_train)).flip(0)
        network = FeatureSelectionNetwork(X_train.shape[1], 32, len(balance)).to(device)
        criterion = nn.CrossEntropyLoss(weight=balance.to(torch.float32).to(device))
    elif mode == "regression":
        network = FeatureSelectionNetwork(X_train.shape[1], 32, 1).to(device)
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    trainloader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size)

    optimizer = optim.Adam(
        [
            {"params": network.fc1.parameters(), "lr": 4e-3},
            {"params": network.cont.parameters(), "lr": 3e-4},
        ]
    )
    temperature = 2.0
    
    if target_features_mode == "auto":
        if X_train.shape[1] > 10000:
            target_features_mode = "raw"
        else:
            target_features_mode = "target"
        
    if target_features_mode == "target":
        target_features = 2
        target_features = target_features / X_train.shape[1]
        
    elif target_features_mode == "raw":
        target_features = None


    for epoch in tqdm.trange(epochs, disable=not verbose):
        total = 0
        correct = 0
        epoch_loss = 0
        for data in trainloader:
            X_, y_ = data
            X_, y_ = X_.to(device), y_.to(device).to(torch.float32)

            optimizer.zero_grad()
            output, selected_no = network(X_.float(), temperature=temperature)
            if mode == "classification":
                proper_y = y_.argmax(axis=1)
            elif mode == "regression":
                proper_y = y_
                
            if target_features is not None:
                loss = criterion(output, proper_y) + fs_balance * (selected_no - target_features) ** 2
            else:
                loss = criterion(output, proper_y) + fs_balance * selected_no
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            predicted = torch.argmax(output, axis=1)
            real = torch.argmax(y_, axis=1)
            total += y_.size(0)
            correct += (predicted == real).sum().item()

            if epoch > epochs:
                temperature = 0
                network.fc1.requires_grad_(False)
                network.emb.requires_grad_(False)
        temperature = temperature * temperature_decay

    features_sum = torch.zeros(X_train.shape[1], dtype=torch.int32)
    for data in trainloader:
        X_, y_ = data
        X_, y_ = X_.to(device), y_.to(device).to(torch.float32)
        features = network.fc1(network.emb)
        features = gumbel_sigmoid(features, tau=0, hard=True)
        features = features.cpu().detach().numpy()
        features_sum += features.sum(axis=0)

    return features_sum.numpy(), network
