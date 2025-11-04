import json
from mlps_shallow import shallow_nn_training
from mlps_medium import medium_nn_training
from mlps_deep import deep_nn_training

print("Running all three architectures for MNIST and CIFAR10")
final_report = {}

final_report["shallow_mnist"] = shallow_nn_training("MNIST")
print(f"shallow_mnist: {final_report['shallow_mnist']}")

final_report["shallow_cifar10"] = shallow_nn_training("CIFAR10")
print(f"shallow_cifar10: {final_report['shallow_cifar10']}")

final_report["medium_mnist"] = medium_nn_training("MNIST")
print(f"medium_mnist: {final_report['medium_mnist']}")

final_report["medium_cifar10"] = medium_nn_training("CIFAR10")
print(f"medium_cifar10: {final_report['medium_cifar10']}")

final_report["deep_mnist"] = deep_nn_training("MNIST")
print(f"deep_mnist: {final_report['deep_mnist']}")

final_report["deep_cifar10"] = deep_nn_training("CIFAR10")
print(f"deep_cifar10: {final_report['deep_cifar10']}")

print(json.dumps(final_report, indent=2))
with open("nn_final_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print("Completed all three architectures for MNIST and CIFAR10")