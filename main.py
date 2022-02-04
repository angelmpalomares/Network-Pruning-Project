from argparse import Namespace
from collections import OrderedDict
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import argparse

batch_size = 64


class Classifier(nn.Module):
    """Classifier that makes predictions on images of animals."""

    def __init__(self, backbone="MLP", device="cuda:0"):
        super(Classifier, self).__init__()
        """Create an untrained classifier.

        Args:
            backbone: the string ("MLP", "CNN" or ResNet) that indicates which backbone network will be used.
            device: the string ("cpu", "cuda:0", "cuda:1", ...) that indicates the device to use.
        """

        # class attributes
        self.num_outputs = 10  # in our problem we have 7 total classes
        self.net = None  # the neural network, composed of backbone + additional projection to the output space
        self.device = torch.device(device)  # the device on which data will be moved
        self.data_mean = torch.zeros(3)  # the mean of the training data on the 3 channels (RGB)
        self.data_std = torch.ones(3)  # the standard deviation of the training data on the 3 channels (RGB)

        # creating the network
        if backbone is not None and backbone == "MLP":

            # Case 1, using an MLP
            self.net = nn.Sequential(OrderedDict([

                ('one', nn.Linear(3072, 1000)),
                ('two', nn.ReLU(inplace=True)),
                ('three', nn.Linear(1000, 500)),
                ('four', nn.ReLU(inplace=True)),
                ('five', nn.Linear(500, 250)),
                ('six', nn.ReLU(inplace=True)),
                ('seven', nn.Linear(250, 100)),
                ('eight', nn.ReLU(inplace=True)),
                ('nine', nn.Linear(100, 50)),
                ('ten', nn.ReLU(inplace=True)),
                ('eleven', nn.Dropout()),
                ('twelve', nn.Linear(50, self.num_outputs))
            ]))

            # moving the network to the right device memory
            self.net.to(self.device)


        elif backbone is not None and backbone == "CNN":

            # Case 2: CNN network
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 256, kernel_size=5, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
                nn.Flatten(),
                nn.Linear(1024, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, self.num_outputs)
            )

            # moving the network to the right device memory
            self.net.to(self.device)

        elif backbone is not None and backbone == "ResNet":

            if device == "cpu":

                self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1)
                self.rel1 = nn.ReLU(inplace=True)
                self.max1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
                self.rel2 = nn.ReLU(inplace=True)
                self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
                self.rel3 = nn.ReLU(inplace=True)
                self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
                self.rel4 = nn.ReLU(inplace=True)
                self.conv5 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
                self.flat = nn.Flatten()
                self.lin1 = nn.Linear(1024, 4096)
                self.rel5 = nn.ReLU(inplace=True)
                self.lin2 = nn.Linear(4096, 4096)
                self.rel6 = nn.ReLU(inplace=True)
                self.drop = nn.Dropout()
                self.output = nn.Linear(4096, self.num_outputs)
            else:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1).cuda()
                self.rel1 = nn.ReLU(inplace=True).cuda()
                self.max1 = nn.MaxPool2d(2).cuda()
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1).cuda()
                self.rel2 = nn.ReLU(inplace=True).cuda()
                self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1).cuda()
                self.rel3 = nn.ReLU(inplace=True).cuda()
                self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0).cuda()
                self.rel4 = nn.ReLU(inplace=True).cuda()
                self.conv5 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0).cuda()
                self.flat = nn.Flatten().cuda()
                self.lin1 = nn.Linear(1024, 4096).cuda()
                self.rel5 = nn.ReLU(inplace=True).cuda()
                self.lin2 = nn.Linear(4096, 4096).cuda()
                self.rel6 = nn.ReLU(inplace=True).cuda()
                self.drop = nn.Dropout().cuda()
                self.output = nn.Linear(4096, self.num_outputs).cuda()

        else:
            if backbone is not None:
                raise ValueError("Unknown backbone: " + str(backbone))
            else:
                raise ValueError("Specify a backbone network!")

        # registering data_mean and data_std as 'buffers', so they will be saved together with the net
        self.net.register_buffer("data_mean_buffer", torch.tensor([0.49139968, 0.48215841, 0.44653091]))
        self.net.register_buffer("data_std_buffer", torch.tensor([0.24703223, 0.24348513, 0.26158784]))

    def save(self, file_name):
        """Save the classifier (network and data mean and std)."""

        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        """Load the classifier (network and data mean and std)."""

        # since our classifier is a nn.Module, we can load it using pytorch facilities (mapping it to the right device)
        self.net.load_state_dict(torch.load(file_name, map_location=self.device))

        # updating data_mean and data_std in order to mach the values in the 'buffers' loaded with the function above
        for name, tensor in self.net.named_buffers():
            if name == "data_mean_buffer":
                self.data_mean[:] = tensor[:]
            elif name == "data_std_buffer":
                self.data_std[:] = tensor[:]

    def forward(self, x):
        """Compute the output of the network."""

        logits = self.net(x)  # outputs before applying the activation function
        outputs = torch.nn.functional.softmax(logits, dim=1)

        # we also return the logits (useful in order to more precisely compute the loss function)
        return outputs, logits

    def forward_res(self, x):

        O1 = self.conv1(x)
        O2 = self.rel1(O1)
        O3 = self.max1(O2)
        O4 = self.conv2(O3) + O1
        O5 = self.rel2(O4)
        O6 = self.conv3(O5)
        O7 = self.rel3(O6)
        O8 = self.conv4(O7) + O6
        O9 = self.rel4(O8)
        O10 = self.conv5(O9)
        O11 = self.flat(O10)
        O12 = self.lin1(O11)
        O13 = self.rel5(O12)
        O14 = self.lin2(O13)
        O15 = self.rel6(O14)
        O16 = self.drop(O15)
        logits = self.output(O16)
        outputs = torch.nn.functional.softmax(logits, dim=1)

        return outputs, logits

    @staticmethod
    def decision(outputs):
        """Given the tensor with the net outputs, compute the final decision of the classifier (class label).

        Args:
            outputs: the 2D tensor with the outputs of the net (each row is about an example).

        Returns:
            1D tensor with the main class IDs (for each example).
        """

        # the decision on main classes is given by the winning class (since they are mutually exclusive)
        main_class_ids = torch.argmax(outputs, dim=1)

        return main_class_ids

    def train_classifier(self, train_set, validation_set, lr, epochs, netname):

        # initializing some elements
        best_val_acc = -1.  # the best accuracy computed on the validation data (main classes)
        best_epoch = -1  # the epoch in which the best accuracy above was computed

        # ensuring the classifier is in 'train' mode (pytorch)
        self.net.train()

        # creating the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr)

        # loop on epochs!
        for e in range(0, epochs):

            # epoch-level stats (computed by accumulating mini-batch stats)
            # accuracy is computed on main classes
            epoch_train_acc = 0.
            epoch_train_loss = 0.
            epoch_num_train_examples = 0

            # stop condition

            for local_batch, local_labels in train_set:

                # counting
                batch_num_train_examples = local_batch.shape[0]
                epoch_num_train_examples += batch_num_train_examples
                # Reshape the data for the expected input of the MLP
                if netname == 'MLP':
                    local_batch = local_batch.reshape(-1, 32 * 32 * 3)

                # moving mini-batch data to the right device
                local_batch = local_batch.to(self.device)
                local_labels = local_labels.to(self.device)

                # computing the network output on the current mini-batch

                outputs, logits = self.forward(local_batch)

                # computing the loss function
                loss = Classifier.__loss(logits, local_labels)

                # computing gradients and updating the network weights
                optimizer.zero_grad()  # zeroing the memory areas that were storing previously computed gradients
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights

                with torch.no_grad():  # keeping these operations out of those for which we will compute the gradient
                    self.net.eval()  # switching to eval mode

                    # computing performance
                    batch_train_acc = self.__performance(outputs, local_labels)

                    # accumulating performance measures to get a final estimate on the whole training set
                    epoch_train_acc += batch_train_acc * batch_num_train_examples

                    # accumulating other stats
                    epoch_train_loss += loss.item() * batch_num_train_examples

                    self.net.train()  # going back to train mode

                    # printing (mini-batch related) stats on screen
                    # print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.2f}".format(loss.item(), batch_train_acc))

            val_acc = self.eval_classifier(validation_set, netname)

            # saving the model if the validation accuracy increases
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1
                self.save("{}.pth".format(netname))

            epoch_train_loss /= epoch_num_train_examples

            # printing (epoch related) stats on screen
            print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.2f}, val_acc={4:.2f}"
                   + (", BEST!" if best_epoch == e + 1 else ""))
                  .format(e + 1, epochs, epoch_train_loss,
                          epoch_train_acc / epoch_num_train_examples, val_acc))

    def eval_classifier(self, data_set, netname):
        """Evaluate the classifier on the given data set."""

        # checking if the classifier is in 'eval' or 'train' mode (in the latter case, we have to switch state)
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()  # enforcing evaluation mode

        # lists on which the mini-batch network outputs (and the targets) will be accumulated
        cpu_batch_outputs = []
        cpu_batch_labels = []

        with torch.no_grad():  # keeping off the autograd engine

            # loop on mini-batches to accumulate the network outputs
            # for local_batch, local_labels in data_set:
            for _, (local_batch, local_labels) in enumerate(data_set):

                if netname == 'MLP':
                    local_batch = local_batch.reshape(-1, 32 * 32 * 3)

                local_batch = local_batch.to(self.device)
                local_labels = local_labels.to(self.device)

                # computing the network output on the current mini-batch
                outputs, _ = self.forward(local_batch)
                cpu_batch_outputs.append(outputs)
                cpu_batch_labels.append(local_labels)

            # computing the performance of the net on the whole dataset
            acc = self.__performance(torch.cat(cpu_batch_outputs, dim=0), torch.cat(cpu_batch_labels, dim=0))

        if training_mode_originally_on:
            self.net.train()  # restoring the training state, if needed

        return acc

    @staticmethod
    def __loss(logits, labels):
        """Compute the loss function of the classifier.

        Args:
            logits: the (partial) outcome of the forward operation.
            labels: 1D tensor with the class labels.

        Returns:
            The value of the loss function.
        """

        tot_loss = F.cross_entropy(logits, labels, reduction="mean")
        return tot_loss

    def __performance(self, outputs, labels):
        """Compute the accuracy in predicting the main classes.

        Args:
            outputs: the 2D tensor with the network outputs for a batch of samples (one example per row).
            labels: the 1D tensor with the expected labels.

        Returns:
            The accuracy in predicting the main classes.
        """

        # taking a decision
        main_class_ids = self.decision(outputs)

        # computing the accuracy on main classes
        right_predictions_on_main_classes = torch.eq(main_class_ids, labels)
        acc_main_classes = torch.mean(right_predictions_on_main_classes.to(torch.float) * 100.0).item()

        return acc_main_classes

    # Basic function to just prune the model depending on the technique
    def prune_net(self, netname, technique, parameter, percentage):

        if netname == 'MLP':
            parameters_to_prune = (
                (self.net.one, parameter),
                (self.net.three, parameter),
                (self.net.five, parameter),
                (self.net.seven, parameter),
                (self.net.nine, parameter),
                (self.net.twelve, parameter))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=technique,
            amount=percentage,
        )

    # Function to measure size of the model
    # def model_size(self):


# Function to test the processing speed and performance of the pruned model
def pruning_performance(net, dataset, model, n):
    # We evaluate the model n times to have a meaningful measurement
    accuracy = 0
    initial_time = time.time()
    for i in range(n):
        accuracy += net.eval_classifier(dataset, model)
    total_time = time.time() - initial_time
    return total_time / n, accuracy / n


def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', choices=['train', 'eval', 'prune'],
                        help='train or evaluate the classifier')
    parser.add_argument('--backbone', type=str, default='MLP', choices=['MLP', 'CNN', 'ResNet'],
                        help='backbone network for feature extraction (default: MLP)"')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (Adam) (default: 0.001)')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cuda:0)')
    parser.add_argument('--pruning', type=str, default='random', choices=['random', 'layer', 'model'],
                        help='pruning technique (default: random)"')
    parser.add_argument('--prune_parameters', type=str, default='weight', choices=['weight', 'bias'],
                        help='parameter to prune (default: weight)"')

    parsed_arguments: Namespace = parser.parse_args()

    return parsed_arguments


if __name__ == "__main__":

    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    if args.mode == 'train':
        print("Training the classifier...")

        # preparing dataset

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        ])

        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(([0.49421428, 0.48513139, 0.45040909]), (0.24665252, 0.24289226, 0.26159238))
        ])

        _train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        # Splitting dataset and creating batches

        _val_set, _test_set = torch.utils.data.random_split(testset, [5000, 5000])

        train_loader = torch.utils.data.DataLoader(_train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=0, pin_memory=False)

        validation_loader = torch.utils.data.DataLoader(_val_set, batch_size=batch_size,
                                                        shuffle=True, num_workers=0, pin_memory=False)

        test_loader = torch.utils.data.DataLoader(_test_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=0, pin_memory=False)

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)

        # training the classifier
        _classifier.train_classifier(train_loader, validation_loader, args.lr, args.epochs, args.backbone)

        # loading the model that yielded the best results in the validation data (during the training epochs)
        print("Training complete, loading the best found model...")
        _classifier.load('{}.pth'.format(args.backbone))

        # computing the performance of the final model in the prepared data splits
        print("Evaluating the classifier...")
        _train_acc = _classifier.eval_classifier(train_loader, args.backbone)
        _val_acc = _classifier.eval_classifier(validation_loader, args.backbone)
        _test_acc = _classifier.eval_classifier(test_loader, args.backbone)

        print("train set:\tacc={0:.2f}".format(_train_acc))
        print("val set:\tacc={0:.2f}".format(_val_acc))
        print("test set:\tacc={0:.2f}".format(_test_acc))

    elif args.mode == 'eval':
        print("Evaluating the classifier...")

        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(([0.49421428, 0.48513139, 0.45040909]), (0.24665252, 0.24289226, 0.26159238))
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        _val_set, _test_set = torch.utils.data.random_split(testset, [5000, 5000])

        test_loader = torch.utils.data.DataLoader(_test_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=0, pin_memory=False)

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)

        # loading the classifier
        _classifier.load('{}.pth'.format(args.backbone))

        # computing the classifier performance
        _acc = _classifier.eval_classifier(test_loader, args.backbone)

        print("acc={0:.2f}".format(_acc))

    elif args.mode == 'prune':

        if args.pruning == 'random':
            technique = prune.RandomUnstructured
        elif args.pruning == 'layer' or 'model':
            technique = prune.L1Unstructured

        # We set the percentages to prune, we will arrive to 60%
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        # In order to prune a model, the desired model is created and loaded
        _classifier = Classifier(args.backbone, args.device)
        _classifier.load('{}.pth'.format(args.backbone))

        for number in percentages:
            _classifier.prune_net(args.backbone, technique, args.prune_parameters, number)

