# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'num_epochs': 100,# Number of epochs to train for.
    'learning_rate': 0.002,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'dataset' : 'MNIST'
                ,# Dataset to use. Choose from {MNIST, FashionMNIST，CIFAR,COIL}. CASE MUST MATCH EXACTLY!!!!!
    'classes' : 0,
    'pro' : 2}