from torchvision import datasets
import torch
import torchvision.transforms as transforms

class Solver():

    def __init__(self, args):

        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_data = self.dataset_selector(args)
        print(self.training_data)

        pass

    def fit(self):
        #     fit model!
        pass




    # ---------------------------- AUXILIARY METHODS-------------------------------

    def dataset_selector(self,args):
        if (args.CIFAR):
            training_data = datasets.CIFAR100(
                root=args.data_root,
                train=True,
                download=True,
                transform=self.data_agumentation(args.data_agumentation))
        else:
            training_data = datasets.CIFAR10(
                root=args.data_root,
                train=True,
                download=True,
                transform=self.data_agumentation(args.data_agumentation))

        return training_data



    def data_agumentation(self,selector):
        transforms_list = list()
        if(selector == 1):
            # add transformations
            pass
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

