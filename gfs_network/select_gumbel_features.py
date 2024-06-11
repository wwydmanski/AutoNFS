from .feature_selection_network import FeatureSelectionNetwork, gumbel_sigmoid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

def select_gumbel_features(X_train, y_train, device="cpu", verbose=False, temperature_decay=0.9999, epochs=300):
    balance = (y_train.sum(axis=0) / len(y_train)).flip(0)

    network = FeatureSelectionNetwork(X_train.shape[1], 16, len(balance)).to(device)

    trainloader = DataLoader(list(zip(X_train, y_train)), batch_size=8)

    criterion = nn.CrossEntropyLoss(weight=balance.to(torch.float32).to(device))
    optimizer = optim.Adam([
        {"params": network.fc1.parameters(), "lr": 3e-6},
        {"params": network.cont.parameters(), "lr": 3e-4},
    ])
    temperature = 2.0

    for epoch in tqdm.trange(epochs, disable=not verbose):
        total = 0
        correct = 0
        epoch_loss = 0
        for data in trainloader:
            X_, y_ = data
            X_, y_ = X_.to(device), y_.to(device).to(torch.float32)

            optimizer.zero_grad()
            output, selected_no = network(X_.float(), temperature=temperature)
            loss = criterion(output, y_.argmax(axis=1)) + selected_no
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            predicted = torch.argmax(output, axis=1)
            real = torch.argmax(y_, axis=1)
            total += y_.size(0)
            correct += (predicted == real).sum().item()
            temperature = temperature * temperature_decay
            
            if epoch > (epochs*2//3):
                temperature = 0
                network.fc1.requires_grad_(False)

    features_sum = torch.zeros(X_train.shape[1], dtype=torch.int32)
    for data in trainloader:
        X_, y_ = data
        X_, y_ = X_.to(device), y_.to(device).to(torch.float32)
        features = network.fc1(X_.float())
        features = gumbel_sigmoid(features, tau=0, hard=True)
        features = features.cpu().detach().numpy()
        features_sum += features.sum(axis=0)
        
    return features_sum.numpy()