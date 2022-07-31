import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import csv
import random
from tqdm import tqdm
from dataloader import get_data
from utils import *
from config import params
from novelty_detector import test

if (params['dataset'] == 'MNIST' or params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Encoder, Discriminator, ZDiscriminator, DHead, QHead
elif (params['dataset'] == 'CIFAR' or params['dataset'] == 'COIL'):
    from models.cifar_model import Generator, Encoder, ZDiscriminator, Discriminator, DHead, QHead

def train(classes):
    params['classes'] = classes

    # Set random seed for reproducibility.
    seed = 1123
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # Use GPU if available.
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    dataloader = get_data(params['dataset'], params['batch_size'], params['classes'], True, params['pro'])

    # Set appropriate hyperparameters depending on the dataset used.
    # The values given in the InfoGAN paper are used.
    # num_z : dimension of latent codes
    # num_dis_c : number of discrete latent code used.
    # dis_c_dim : dimension of discrete latent code.
    if (params['dataset'] == 'MNIST' or params['dataset'] == 'FashionMNIST'):
        params['num_z'] = 16
        params['num_dis_c'] = 1
        params['dis_c_dim'] = 4
    elif (params['dataset'] == 'CIFAR' or params['dataset'] == 'COIL'):
        params['num_z'] = 16
        params['num_dis_c'] = 1
        params['dis_c_dim'] = 4
    # Plot the training images.
    sample_batch = next(iter(dataloader))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(
        sample_batch[0].to(device)[: 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('results/Training Images {}'.format(params['dataset']))
    plt.close('all')

    # Initialise the network.
    netE = Encoder().to(device)
    netE.apply(weights_init)
    print(netE)

    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    print(discriminator)

    ZD = ZDiscriminator().to(device)
    ZD.apply(weights_init)
    print(ZD)

    netD = DHead().to(device)
    netD.apply(weights_init)
    print(netD)

    netQ = QHead().to(device)
    netQ.apply(weights_init)
    print(netQ)

    # Loss for reconstruction
    MSE_Loss = nn.MSELoss()
    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ_dis = nn.CrossEntropyLoss()


    # Adam optimiser is used.
    optimD = optim.RMSprop([{'params': discriminator.parameters()}, {'params': netD.parameters()}],
                           lr=params['learning_rate'])
    optimG = optim.RMSprop([{'params': netG.parameters()}, {'params': netQ.parameters()}],
                           lr=params['learning_rate'])
    optimZD = optim.Adam([{'params': ZD.parameters()}], lr=params['learning_rate'],
                         betas=(params['beta1'], params['beta2']))
    optimAE = optim.Adam([{'params': netG.parameters()}, {'params': netE.parameters()}], lr=params['learning_rate'],
                         betas=(params['beta1'], params['beta2']))

    # Fixed Noise
    z = torch.randn(100, params['num_z'], 1, 1, device=device)
    fixed_noise = z
    if (params['num_dis_c'] != 0):
        idx = np.arange(params['dis_c_dim']).repeat(25)

        dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)

        for i in range(params['num_dis_c']):
            dis_c[torch.arange(0, 100), i, idx] = 1.0

        dis_c = dis_c.view(100, -1, 1, 1)

        fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

    real_label = 1
    fake_label = 0

    # List variables to store results pf training.
    img_list = []
    G_losses = []
    D_losses = []

    print("-" * 25)
    print("Starting Training Loop...\n")
    print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (
        params['num_epochs'], params['batch_size'], len(dataloader)))
    print("-" * 25)

    start_time = time.time()
    iters = 0

    re = 0
    re_all = 0
    ep = 0
    ep_all = 0

    loss_head = ['Loss_D', 'Loss_G', 'Loss_Info', 'Loss_E', 'Loss_RE']
    results_head = ['result_x','result_z','result_fd','result_noz', 'result']
    loss = []
    results = []

    for epoch in range(params['num_epochs']):
        epoch_start_time = time.time()

        for i, (data, in_label, r_label) in enumerate(tqdm(dataloader), 0):
            list_loss = []

            # Get batch size
            b_size = data.size(0)
            # Transfer data tensor to GPU/CPU (device)
            input = data.to(device)
            # Get one_hot label
            one_hot = torch.zeros(b_size, params['num_dis_c'], params['dis_c_dim'], device=device)

            for i in range(params['num_dis_c']):
                one_hot[torch.arange(0, b_size), i, in_label] = 1.0
            one_hot = one_hot.view(b_size, -1, 1, 1)

            ########################### Updating discriminator and DHead ###########################
            optimD.zero_grad()
            # Real data
            label = torch.full((b_size,), real_label, dtype=torch.float32, device=device)
            output1 = discriminator(input)
            probs_real = netD(output1).view(-1)
            loss_real = criterionD(probs_real,label)
            loss_real.backward()

            # Fake data
            label.fill_(fake_label)
            noise = torch.randn((b_size, params['num_z'])).view(b_size, params['num_z'], 1, 1).to(device)
            noise = torch.cat((noise, one_hot), dim=1)

            fake_data = netG(noise)
            output2 = discriminator(fake_data.detach())
            probs_fake_s = netD(output2).view(-1)
            loss_fake_s = criterionD(probs_fake_s,label)
            loss_fake_s.backward()

            # Net Loss for the discriminator
            D_loss = loss_real + loss_fake_s

            # Update parameters
            optimD.step()

            list_loss.append(D_loss.item())

            ########################### Updating Generator and QHead ###########################
            optimG.zero_grad()

            # Fake data of sample
            output_s = discriminator(fake_data)

            # Fake data of z,it is also reconstruction(x_d)
            z = torch.cat((netE(input), one_hot), dim=1)
            x_d = netG(z)
            # output_x = discriminator(input)
            output_z = discriminator(x_d)

            q_logits_s, _, _ = netQ(output_s)
            q_logits_z, _, _ = netQ(output_z)
            target = torch.LongTensor(in_label).to(device)

            label.fill_(real_label)
            probs_fake = netD(output_s).view(-1)
            gen_loss_s = criterionD(probs_fake,label)
            

            # Calculating loss for discrete latent code.
            dis_loss_s = criterionQ_dis(q_logits_s, target)
            dis_loss_z = criterionQ_dis(q_logits_z, target)
            dis_loss = dis_loss_s + dis_loss_z

            # Net loss for generator.
            G_loss = gen_loss_s + dis_loss
            # Calculate gradients.
            G_loss.backward()
            # Update parameters.
            optimG.step()

            ########################### Updating ZDiscriminator ###########################
            optimZD.zero_grad()

            noise = torch.randn((b_size, params['num_z'])).to(device)
            noise = noise.requires_grad_(True)
            label.fill_(real_label)

            ZD_result = ZD(noise).view(-1)
            ZD_real_loss = criterionD(ZD_result, label)
            ZD_real_loss.backward()

            label.fill_(fake_label)
            z = netE(input).squeeze().detach()
            ZD_result_fake = ZD(z).view(-1)
            ZD_fake_loss = criterionD(ZD_result_fake, label)

            ZD_fake_loss.backward()

            optimZD.step()

            ########################### Updating AE ###########################
            optimAE.zero_grad()

            label.fill_(real_label)
            z = netE(input)
            ZD_result = ZD(z.squeeze()).squeeze()
            E_loss = criterionD(ZD_result, label)

            z = torch.cat((z, one_hot), dim=1)
            x_d = netG(z)

            Recon_loss = 2 * MSE_Loss(input, x_d)

            (E_loss + Recon_loss).backward()

            optimAE.step()
            list_loss.append(E_loss.item())
            list_loss.append(Recon_loss.item())
            ########################### Updating over ###########################

            # Check progress of training.

            # Save the losses for plotting.
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())

            iters += 1

        epoch_time = time.time() - epoch_start_time

        loss.append([D_loss.item(), G_loss.item(), dis_loss.item(), E_loss.item(), Recon_loss.item()])

        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_Info: %.4f\tLoss_E: %.4f\tLoss_RE: %.4f'
              % (epoch + 1, params['num_epochs'], i, len(dataloader),
                 D_loss.item(), G_loss.item(), dis_loss.item(), E_loss.item(), Recon_loss.item()))
        print("Time taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
        # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

        # Generate image to check performance of generator.
        if (epoch!=0):
            with torch.no_grad():
                gen_data = netG(fixed_noise).detach().cpu()
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
            plt.savefig("results/Epoch_%d {}".format(params['dataset']) % (epoch))
            print("Epoch_%d {}".format(params['dataset']) % (epoch))
            plt.close('all')

        # Save network weights.
        torch.save({
            'netE': netE.state_dict(),
            'netG': netG.state_dict(),
            'discriminator': discriminator.state_dict(),
            'netD': netD.state_dict(),
            'netQ': netQ.state_dict(),
            'params': params
        }, 'checkpoint/model_%d_{}'.format(params['classes']) % (epoch))

        test_r, test_r_all, roc_auc = test(epoch)
        results.append(roc_auc)
        if re <= test_r:
            if epoch > 20:
                ep = epoch
                re = test_r
        if re_all <= test_r_all:
            if epoch > 20:
                ep_all = epoch
                re_all = test_r_all

    # 把损失写进csv文件中
    with open('results/loss.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(loss_head)
        for i in range(+1):
            writer.writerow(loss[i])

    # 把results写进csv文件中
    with open('results/results_{}.csv'.format(params['classes']), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results_head)
        for i in range(epoch+1):
            writer.writerow(results[i])

    training_time = time.time() - start_time
    print("-" * 50)
    print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
    print("-" * 50)

    print("best result:" + str(re))
    print("best epoch:" + str(ep))
    print("best result_all:" + str(re_all))
    print("best epoch:" + str(ep_all))


    # Save network weights.
    torch.save({
        'netE': netE.state_dict(),
        'netG': netG.state_dict(),
        'discriminator': discriminator.state_dict(),
        'netD': netD.state_dict(),
        'netQ': netQ.state_dict(),
        'params': params
    }, 'checkpoint/model_final_{}'.format(params['dataset']))

    # Animation showing the improvements of the generator.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    anim.save('results/infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')


train(3)

