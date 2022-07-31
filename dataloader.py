import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from random import shuffle
from PIL import Image

# Directory containing the data.
root = 'data/'

def tensor2PIL(tensor):  # tensor-> PIL
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image
    image = unloader(image)
    return image


def PIL2tensor(img):
    loader = transforms.Compose([
        transforms.ToTensor()
    ])
    image = loader(img)
    return image.to(torch.float)

def rotate(dataset):
    data = []
    rot_label = []
    real_label = []
    for i in range(4):
        for x in dataset:
            x_i = transforms.functional.rotate (tensor2PIL(x[0]), i * 90)

            data.append(PIL2tensor(x_i))
            rot_label.append(i)
            real_label.append(x[1])

    dataset = [(im,rotl,reall) for im,rotl,reall in zip(data,rot_label,real_label)]
    shuffle(dataset)
    return dataset



def get_data(dataset, batch_size, classes, is_train, pro= 2):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset_tr = dsets.MNIST(root+'mnist/', train=True,
                                download=True, transform=transform)

        dataset_te = dsets.MNIST(root + 'mnist/', train=False,
                                 download=True, transform=transform)

    elif dataset == 'CIFAR':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
            ])

        dataset_tr = dsets.CIFAR10(root + 'cifar/', train=True,
                                 download=True, transform=transform)

        dataset_te = dsets.CIFAR10(root + 'cifar/', train=False,
                                 download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset_tr = dsets.FashionMNIST(root+'fashionmnist/', train=True,
                                download=True, transform=transform)

        dataset_te = dsets.FashionMNIST(root + 'fashionmnist/', train=False,
                                 download=True, transform=transform)

    elif dataset == 'COIL':
        data = []
        label = []
        transform = transforms.Compose ([
            transforms.Resize (32),
            transforms.CenterCrop (32),
            transforms.ToTensor ()
        ])
        for l in range(100):
            for i in range(72):
                img = Image.open(root + 'COIL/obj' + str(l+1) + '__' + str(i*5) + '.png' )
                x_i = transform(img)
                data.append(x_i)
                label.append(l)

        dataset_all = [(im,label) for im,label in zip(data,label)]



    # choose the class we want and get rotate
    if pro==1:
        if dataset == 'COIL':
            da = dataset_all
        else:
            da = dataset_tr + dataset_te
        data_in = [x for x in da if x[1] == classes]
        data_out = [x for x in da if x[1] != classes]
        data_train = data_in[:int(len(data_in) * 0.8)]
        data_test = data_in[int(len(data_in) * 0.8):]
        if is_train:
            dataset = rotate(data_train)
        else:
            dataset = data_test + random.sample(data_out, len(data_test))

    elif pro==2:
        if is_train:
            dataset_tr = [x for x in dataset_tr if x[1] == classes]
            dataset = rotate(dataset_tr)
        else:
            dataset = dataset_te

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    return dataloader

