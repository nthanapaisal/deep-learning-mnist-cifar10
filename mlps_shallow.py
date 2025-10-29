# MLPS
# archs 1. Shallow (1 hidden layer, e.g., 128 units) 2. Medium-depth (3 hidden layers, e.g., [512, 256, 128]) 3. Deep (at least 5 hidden layers, your choice)

import os
import time
import random
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from preprocessing import load_data


torch.manual_seed(42)
random.seed(42)

report = {}
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print("Using device:", device)

# ∗ Learning rate (e.g. {0.01, 0.001, 0.0001})
# ∗ Batch size (e.g. {32, 64, 128})
# ∗ Optimizer (SGD vs. Adam)
# ∗ Dropout rate (e.g. {0.2,0.5})

# nn class Shallow
# https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, img_size, dropout_rate, in_channels_thickness=1): # W * H * T
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_channels_thickness * img_size * img_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, data):
        flatten_data = self.flatten(data)
        logits = self.linear_relu_stack(flatten_data)
        return logits


# Shallow
def shallow_nn_training(dataset_name):
    max_acc = float('-inf')
    winner = {}
    start_time = time.time()
    for batch_size in [64, 128]:
        # Load and Split Data
        mnist_train, mnist_val, mnist_test, cifar10_train, cifar10_val, cifar10_test = load_data(batch_size)
        if dataset_name == "MNIST":
            input_train, input_val, input_test = mnist_train, mnist_val, mnist_test
            pixel_size = 28
            in_channels_thickness = 1
        else:
            input_train, input_val, input_test = cifar10_train, cifar10_val, cifar10_test
            pixel_size = 32
            in_channels_thickness = 3

        # https://docs.pytorch.org/docs/stable/optim.html#module-torch.optim
        for optimizer_name in ["sgd","adam"]:
            for lr in [0.01, 0.001, 0.0001]:
                for dropout_rate in [0.2, 0.5]:

                    # nn Shallow model
                    model = ShallowNeuralNetwork(pixel_size, dropout_rate, in_channels_thickness).to(device)

                    # loss func
                    cross_entropy_loss = nn.CrossEntropyLoss()

                    if optimizer_name == "sgd":
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                    else:
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    model.train()

                    print(f"Dataset: {dataset_name} Hyperparams: batch_size: {batch_size}, optimizer: {optimizer_name}, lr: {lr}, dropout_rate: {dropout_rate}")

                    early_stop_counter = 0
                    best_val_acc_for_this_config = 0.0

                    for epoch in range(30):
                        
                        total_loss = 0.0
                        # train 
                        for images, labels in input_train:
                            # put data in device
                            images, labels = images.to(device), labels.to(device)

                            # forward pass
                            logits = model(images)
                            loss = cross_entropy_loss(logits, labels)

                            # back propagation https://docs.pytorch.org/docs/stable/optim.html#taking-an-optimization-step
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()

                        # validation 
                        model.eval()
                        total_val_loss = 0.0
                        total_samples_nums = 0
                        correct = 0

                        # disable gradient computation for evaluation
                        with torch.no_grad():  
                            for images_val, labels_val in input_val:
                                images_val, labels_val = images_val.to(device), labels_val.to(device)

                                logits_val = model(images_val)
                                val_loss = cross_entropy_loss(logits_val, labels_val)
                                total_val_loss += val_loss.item()

                                total_samples_nums += labels_val.size(0)
                                predicted_index = logits_val.argmax(dim=1)
                                correct += (predicted_index == labels_val).sum().item()
                        
                        # calculate accuracy and update best val acc within these epoch 
                        avg_train_loss = total_loss / len(input_train)
                        avg_val_loss = total_val_loss / len(input_val)
                        val_accuracy = correct / total_samples_nums
                        print(f"epoch: {epoch}, avg_train_loss: {avg_train_loss}, avg_val_loss: {avg_val_loss}")

                        if val_accuracy > best_val_acc_for_this_config:
                            best_val_acc_for_this_config = val_accuracy
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= 3: break

                    # update global acc if the set of hyperpams has the better acc
                    if best_val_acc_for_this_config > max_acc:
                        max_acc = best_val_acc_for_this_config
                        winner = {
                            "accuracy": best_val_acc_for_this_config,
                            "avg_val_loss": total_val_loss / len(input_val),
                            "batch_size": batch_size,
                            "optimizer": optimizer_name,
                            "lr": lr,
                            "dropout_rate": dropout_rate
                        }
                    
    end_time = time.time()
    runtime_min = round((end_time - start_time) / 60, 2)

    print(f"Hypertuning winner: {winner}, dataset: {dataset_name}")

    # final training
    model_final = ShallowNeuralNetwork(pixel_size, winner["dropout_rate"], in_channels_thickness).to(device)

    # loss func
    cross_entropy_loss = nn.CrossEntropyLoss()

    if winner["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model_final.parameters(), lr=winner["lr"], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model_final.parameters(), lr=winner["lr"])

    model_final.train()

    # combine train and val data 
    combined_train = ConcatDataset([input_train.dataset, input_val.dataset])
    combined_train_val = DataLoader(combined_train, batch_size=winner["batch_size"], shuffle=True)

    for epoch in range(10):
        total_loss = 0.0
        for images, labels in combined_train_val:
            # put data in device
            images, labels = images.to(device), labels.to(device)

            # forward pass
            logits = model_final(images)
            loss = cross_entropy_loss(logits, labels)

            # back propagation https://docs.pytorch.org/docs/stable/optim.html#taking-an-optimization-step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # calculate this set of hyperparams perf
    model_final.eval()
    total_test_loss = 0.0
    total_samples_nums = 0
    correct = 0
    # Disable gradient computation for evaluation
    with torch.no_grad():  
        for images_test, labels_test in input_test:
            images_test, labels_test = images_test.to(device), labels_test.to(device)

            logits_test = model_final(images_test)
            test_loss = cross_entropy_loss(logits_test, labels_test)
            total_test_loss += test_loss.item()

            # Calculate accuracy
            total_samples_nums += labels_test.size(0)
            predicted_index = logits_test.argmax(dim=1)
            correct += (predicted_index == labels_test).sum().item()

    accuracy = correct / total_samples_nums
    return {
        "runtime": runtime_min,
        "accuracy": accuracy,
        "avg_train_loss": total_loss / len(combined_train_val),
        "avg_val_loss": winner["avg_val_loss"],
        "avg_test_loss": total_test_loss / len(input_test),
        "batch_size": winner["batch_size"],
        "optimizer": winner["optimizer"],
        "lr": winner["lr"],
        "dropout_rate": winner["dropout_rate"]
    }
