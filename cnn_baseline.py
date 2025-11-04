import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.nn.functional as F

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    mnist_training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    mnist_test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    img, label = mnist_training_data[0]
    print(img.shape)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar10_training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    cifar10_test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    img, label = cifar10_training_data[0]
    print(img.shape)

    mnist_train, mnist_val = random_split(mnist_training_data, [50000, 10000])
    mnist_training_data_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_val_data_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
    mnist_test_data_loader = DataLoader(mnist_test_data, batch_size=batch_size, shuffle=False)

    cifar10_train, cifar10_val = random_split(cifar10_training_data, [45000, 5000])
    cifar10_training_data_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    cifar10_val_data_loader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=False)
    cifar10_test_data_loader = DataLoader(cifar10_test_data, batch_size=batch_size, shuffle=False)

    return ( mnist_training_data_loader, mnist_val_data_loader, mnist_test_data_loader, 
        cifar10_training_data_loader, cifar10_val_data_loader, cifar10_test_data_loader )



import os
import time
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

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

# cnn  (2 convolutional layers + pooling, fully connected layer)
class CnnBaseline(nn.Module):
    def __init__(self, img_size, in_channels_thickness=1):
        super().__init__()
        filter_nums = 32
        filter_size = 3
        window_size, stride = 2, 2
        class_nums = 10
        #MINST 1x28x28 -> 32x28x28 -> 32x14x14 -> 64x14x14 -> 64x7x7
        self.convo_layer1 = nn.Conv2d(in_channels_thickness, filter_nums, filter_size, padding=1)
        self.convo_layer2 = nn.Conv2d(filter_nums, filter_nums*2, filter_size, padding=1)
        self.pool = nn.MaxPool2d(window_size,stride) # pool after each convo layer: compressed spatial details that is not needed
        self.function1 = nn.Linear((filter_nums*2)*(img_size//4)*(img_size//4), 128) # 3136, 128
        self.function2 = nn.Linear(128, class_nums) #128, 10

    def forward(self, data):
        res = self.pool(F.relu(self.convo_layer1(data))) # fully connected hidden layer 
        res = self.pool(F.relu(self.convo_layer2(res))) # output layer into 10 classes
        res = res.view(res.size(0), -1) #flatten to 2D vector [batch size and number of features L*W*T]
        res = F.relu(self.function1(res))
        res = self.function2(res)
        return res

def cnn_baseline_training(dataset_name):
    max_acc = float('-inf')
    winner = {}
    start_time = time.time()
    for batch_size in [64, 128]:
        mnist_train, mnist_val, mnist_test, cifar10_train, cifar10_val, cifar10_test = load_data(batch_size)
        if dataset_name == "MNIST":
            input_train, input_val, input_test = mnist_train, mnist_val, mnist_test
            pixel_size = 28
            in_channels_thickness = 1
        else:
            input_train, input_val, input_test = cifar10_train, cifar10_val, cifar10_test
            pixel_size = 32
            in_channels_thickness = 3

        for optimizer_name in ["sgd","adam"]:
            for lr in [0.01, 0.001, 0.0001]:
                model = CnnBaseline(pixel_size, in_channels_thickness).to(device)

                cross_entropy_loss = nn.CrossEntropyLoss()

                if optimizer_name == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                model.train()

                print(f"Dataset: {dataset_name}, Hyperparams: batch_size: {batch_size}, optimizer: {optimizer_name}, lr: {lr}")

                early_stop_counter = 0
                best_val_acc_for_this_config = 0.0
                val_acc = []

                for epoch in range(30):
                    
                    total_loss = 0.0

                    for images, labels in input_train:

                        images, labels = images.to(device), labels.to(device)

                        logits = model(images)
                        loss = cross_entropy_loss(logits, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    model.eval()
                    total_val_loss = 0.0
                    total_samples_nums = 0
                    correct = 0

                    with torch.no_grad():  
                        for images_val, labels_val in input_val:
                            images_val, labels_val = images_val.to(device), labels_val.to(device)

                            logits_val = model(images_val)
                            val_loss = cross_entropy_loss(logits_val, labels_val)
                            total_val_loss += val_loss.item()

                            total_samples_nums += labels_val.size(0)
                            predicted_index = logits_val.argmax(dim=1)
                            correct += (predicted_index == labels_val).sum().item()
                    
                    avg_train_loss = total_loss / len(input_train)
                    avg_val_loss = total_val_loss / len(input_val)
                    val_accuracy = correct / total_samples_nums
                    val_acc.append(val_accuracy)
                    print(f"epoch: {epoch}, avg_train_loss: {avg_train_loss}, avg_val_loss: {avg_val_loss}")

                    if val_accuracy > best_val_acc_for_this_config:
                        best_val_acc_for_this_config = val_accuracy
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= 3: break

                std_val_acc = np.std(val_acc, ddof=1)

                if best_val_acc_for_this_config > max_acc:
                    max_acc = best_val_acc_for_this_config
                    winner = {
                        "accuracy": best_val_acc_for_this_config,
                        "std_val_acc": std_val_acc,
                        "avg_val_loss": total_val_loss / len(input_val),
                        "batch_size": batch_size,
                        "optimizer": optimizer_name,
                        "lr": lr
                    }
                
    end_time = time.time()
    runtime_min = round((end_time - start_time) / 60, 2)

    print(f"Hypertuning winner: {winner}")

    # final training
    model_final = CnnBaseline(pixel_size, in_channels_thickness).to(device)

    # loss func
    cross_entropy_loss = nn.CrossEntropyLoss()

    if winner["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model_final.parameters(), lr=winner["lr"], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model_final.parameters(), lr=winner["lr"])

    model_final.train()

    combined_train = ConcatDataset([input_train.dataset, input_val.dataset])
    combined_train_val = DataLoader(combined_train, batch_size=winner["batch_size"], shuffle=True)

    for epoch in range(10):
        total_loss = 0.0
        for images, labels in combined_train_val:

            images, labels = images.to(device), labels.to(device)

            logits = model_final(images)
            loss = cross_entropy_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    model_final.eval()
    total_test_loss = 0.0
    total_samples_nums = 0
    correct = 0

    with torch.no_grad():  
        for images_test, labels_test in input_test:
            images_test, labels_test = images_test.to(device), labels_test.to(device)

            logits_test = model_final(images_test)
            test_loss = cross_entropy_loss(logits_test, labels_test)
            total_test_loss += test_loss.item()

            total_samples_nums += labels_test.size(0)
            predicted_index = logits_test.argmax(dim=1)
            correct += (predicted_index == labels_test).sum().item()

    accuracy = correct / total_samples_nums
    return {
        "runtime": runtime_min,
        "accuracy": accuracy,
        "avg_train_loss": total_loss / len(combined_train_val),
        "avg_val_loss": winner["avg_val_loss"],
        "val_acc": winner["accuracy"],
        "std_val_acc": winner["std_val_acc"],
        "avg_test_loss": total_test_loss / len(input_test),
        "batch_size": winner["batch_size"],
        "optimizer": winner["optimizer"],
        "lr": winner["lr"]
    }

print("Running Baseline architecture for MNIST and CIFAR10")
final_report = {}

final_report["cnn_baseline_mnist"] = cnn_baseline_training("MNIST")
print(f"cnn_baseline_mnist: {final_report['cnn_baseline_mnist']}")

final_report["cnn_baseline_cifar10"] = cnn_baseline_training("CIFAR10")
print(f"cnn_baseline_cifar10: {final_report['cnn_baseline_cifar10']}")

print(json.dumps(final_report, indent=2))
with open("cnn_baseline_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print("Completed Baseline architecture for MNIST and CIFAR10")
