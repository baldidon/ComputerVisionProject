import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import multiprocessing as mp
import torch
import torchvision.transforms as transforms
from progetto_cv.scripts.net import Net
import matplotlib.pyplot as plt


class Solver():

    def __init__(self, args):
        self.args = args
        # turn on the CUDA if available. In my case must be available!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.training_data, self.test_data = self.dataset_selector(args)

        self.train_loader = DataLoader(dataset=self.training_data,
                                       batch_size=args.batch_size,
                                       num_workers=mp.cpu_count()-1,
                                       shuffle=True,
                                       drop_last=True)

        self.test_loader = DataLoader(dataset=self.test_data,
                                       batch_size=args.batch_size,
                                       num_workers=mp.cpu_count() - 1,
                                       shuffle=True,
                                       drop_last=True)

        self.net = Net(args.CIFAR).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = self.optimizer_selector(args.optimizer)



    def fit(self):
        args = self.args
        loss_values = list()
        train_acc_values = np.zeros(args.max_epochs)
        test_acc_values = np.zeros(args.max_epochs)

        # handle accuracy plot
        if args.plot == 1:
            plt.ion()
            figure, ax = plt.subplots(figsize=(10, 8))
            x = np.linspace(0,args.max_epochs,1)
            train_y = train_acc_values
            test_y = test_acc_values
            train_plot, = ax.plot(x,train_y)
            test_plot, = ax.plot(x, test_y)
            plt.title("Train and Test accuracy", fontsize=20)
            plt.xlabel("epoch")
            plt.ylabel("accuracy")

        for epoch in range(args.max_epochs):
            self.net.train()  # i wanna use dropout layer, so i this need to activate layer# !

            for step, inputs in enumerate(self.train_loader):
                # load batch in device (gpu)
                images = inputs[0].to(self.device)
                true_classes = inputs[1].to(self.device)

                # forward propagation
                pred_classes = self.net(images)
                loss = self.loss_fn(pred_classes, true_classes)
                loss_values.append(loss)

                # backward propagation
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # accuracy
                train_acc_values[epoch] = self.evaluate(self.training_data)
                test_acc_values[epoch] = self.evaluate(self.test_data)

                if args.plot == 1:
                    train_plot.set_ydata(train_acc_values)
                    test_plot.set_ydata(train_acc_values)
                    figure.canvas.draw()
                    figure.canvas.flush_events()

                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                      format(epoch + 1, args.max_epochs, loss.item(),
                             train_acc_values[epoch], test_acc_values[epoch]))




    # ---------------------------- AUXILIARY METHODS-------------------------------
    def dataset_selector(self,args):
        if (args.CIFAR == 100):
            training_data = datasets.CIFAR100(
                root=args.data_root,
                train=True,
                download=True,
                transform=self.data_agumentation(args.data_agumentation))
            test_data = datasets.CIFAR100(
                root=args.data_root,
                train=False,
                transform=self.data_agumentation(0)
            )
        else:
            training_data = datasets.CIFAR10(
                root=args.data_root,
                train=True,
                download=True,
                transform=self.data_agumentation(args.data_agumentation))
            test_data = datasets.CIFAR10(
                root=args.data_root,
                train=False,
                transform=self.data_agumentation(0)
            )

        return training_data, test_data


    def data_agumentation(self,selector):
        transforms_list = list()
        if selector == 1:
            # transforms_list.append(transforms.ColorJitter(
            #
            # ))
            pass
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(*transforms_list)

    def optimizer_selector(self, selector:str):
        if selector == 'adam' :
            optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.args.lr,
                                         eps=1e-8, betas=(0.9, 0.999), weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=self.args.lr,
                                         eps=1e-8, betas=(0.9, 0.999), weight_decay=self.args.wd)

        return optimizer

    def evaluate(self, data):
        args = self.args
        loader = DataLoader(data,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=False)

        self.net.eval()
        num_correct, num_total = 0, 0

        with torch.no_grad():
            for inputs in loader:
                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device)

                outputs = self.net(images)
                _, preds = torch.max(outputs.detach(), 1)

                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total