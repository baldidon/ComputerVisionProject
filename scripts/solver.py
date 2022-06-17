import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import multiprocessing as mp
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from progetto_cv.scripts.net import Net


class Solver():

    def __init__(self, args):
        print(mp.cpu_count())
        self.args = args
        # turn on the CUDA if available. In my case must be available!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.training_data, self.test_data = self.dataset_selector(args)

        self.train_loader = DataLoader(dataset=self.training_data,
                                       batch_size=args['batch_size'],
                                       num_workers=mp.cpu_count() - 2,
                                       shuffle=True,
                                       drop_last=True)

        self.test_loader = DataLoader(dataset=self.test_data,
                                      batch_size=args['batch_size'],
                                      num_workers=mp.cpu_count() - 2,
                                      shuffle=True,
                                      drop_last=True)

        if args['transfer_learning'] == False:
            self.net = Net(args['CIFAR'], args['transfer_learning']).to(self.device)
        else:
            # TODO
            # self.net = Net(args['CIFAR'] ,args['transfer_learning']).to(self.device)
            # self.net.load_state_dict(torch.load(args['model_root']))
            # # freeze current model
            # for param in self.net.parameters():
            #     param.requires_grad = False
            # # swap classifier, with rrequires_grad = True
            # self.net.fc = nn.Linear( 8 * 8 *256, args['CIFAR'])
            pass

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = self.optimizer_selector(args['optimizer'])

    def fit(self):
        args = self.args
        loss_values = list()
        self.train_acc_values = list()
        self.test_acc_values = list()

        for epoch in range(args['max_epochs']):
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
            self.train_acc_values.append(self.evaluate(self.training_data))
            self.test_acc_values.append(self.evaluate(self.test_data))

            print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                  format(epoch + 1, args['max_epochs'], loss.item(),
                         self.train_acc_values[epoch], self.test_acc_values[epoch]))

            if self.stop_condition(epoch) == True:
                self.stop = True
                epoch_break = epoch
                break

        if (self.stop) == False:
            epoch_break = args['max_epochs']

        # plot accuracy
        if (args['plot'] == 1):
            x = np.linspace(1, epoch_break, epoch_break)
            y_train = self.train_acc_values
            y_test = self.test_acc_values
            plt.plot(x, y_train, color='b', label='train_acc')
            plt.plot(x, y_test, color='r', label='test_acc')
            plt.title('andamento accuracy')
            plt.legend()
            plt.xlabel('epoche')
            plt.ylabel('accuracy')
            plt.show()

        # for further usage, if train was CIFAR-10, save net parameter
        torch.save(self.net.state_dict(), f"model_{args['CIFAR']}")

    # ---------------------------- AUXILIARY METHODS-------------------------------
    def dataset_selector(self, args):
        if args['CIFAR'] == 100:
            training_data = datasets.CIFAR100(
                root=args['data_root'],
                train=True,
                download=True,
                transform=self.data_agumentation(args['data_agumentation']))
            test_data = datasets.CIFAR100(
                root=args['data_root'],
                train=False,
                transform=self.data_agumentation(0)
            )
        else:
            training_data = datasets.CIFAR10(
                root=args['data_root'],
                train=True,
                download=True,
                transform=self.data_agumentation(args['data_agumentation']))
            test_data = datasets.CIFAR10(
                root=args['data_root'],
                train=False,
                transform=self.data_agumentation(0)
            )

        return training_data, test_data

    def data_agumentation(self, selector):
        transforms_list = list()
        if selector == 1:
            transforms_list.append(transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1))
            #  transforms_list.append(transforms.GaussianBlur(kernel_size=(3,3))) # provare senza
            transforms_list.append(transforms.RandomVerticalFlip(p=self.args['p']))
            transforms_list.append(transforms.RandomHorizontalFlip(p=self.args['p']))

        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

    def optimizer_selector(self, selector: str):
        if selector == 'Adam':
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.net.parameters()),
                                         lr=self.args['lr'],
                                         eps=1e-8, betas=(0.9, 0.999), weight_decay=self.args['wd'])
        else:
            optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, self.net.parameters()),
                                          lr=self.args['lr'],
                                          eps=1e-8, betas=(0.9, 0.999), weight_decay=self.args['wd'])

        return optimizer

    def stop_condition(self, epochs: int):
        # check if test acc decreasing for 3 epochs while train acc doesn't
        eps = 0.1
        self.stop = False
        if epochs > 3:
            if ((self.test_acc_values[epochs - 1] - self.test_acc_values[epochs]) > eps) and (
                    self.test_acc_values[epochs - 2] - self.test_acc_values[epochs - 1]) > eps and (
                    self.test_acc_values[epochs - 3] - self.test_acc_values[epochs - 2]) > eps:
                if ((self.train_acc_values[epochs - 1] - self.train_acc_values[epochs]) < -eps) and (
                        self.train_acc_values[epochs - 2] - self.train_acc_values[epochs - 1]) < -eps and (
                        self.train_acc_values[epochs - 3] - self.train_acc_values[epochs - 2]) < -eps:
                    self.stop = True

        return self.stop

    def evaluate(self, data):
        args = self.args
        loader = DataLoader(data,
                            batch_size=args['batch_size'],
                            num_workers=2,
                            shuffle=False)

        self.net.eval()  # to deactivate dropout layer
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
