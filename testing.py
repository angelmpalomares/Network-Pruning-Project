import torch
import torchvision
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.transforms as transforms


transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    ])




train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train
        )
test = torchvision.datasets.CIFAR10(
     root='./data', train=False, download=True,transform=transform_train)

train_loader = torch.utils.data.DataLoader(train, batch_size=64,
                                           shuffle=True, num_workers=0, pin_memory=False)
train_loader=train_loader.reshape(-1, 32 * 32 * 3)
validation_loader = torch.utils.data.DataLoader(test, batch_size=64,
                                                shuffle=True, num_workers=0, pin_memory=False)
for (local_batch, local_labels) in validation_loader:
    images = local_batch.reshape(-1, 32 * 32 * 3)
    print('hola')


a=4







