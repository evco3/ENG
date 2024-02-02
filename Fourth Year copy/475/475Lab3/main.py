import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
import AdaIN_net as net
import custom_dataset as cd
import matplotlib.pyplot as plt
import datetime


def train(n_epochs, n_batches, optimizer, model, train_dataloader, loss_fn, scheduler,
          device, output_plot_path, param_file):
    model.to(device)
    model.train()

    # Empty cache if using cuda GPU as the device
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Initialize loss
    losses_train = []


    for epoch in range(n_epochs):
        epoch_losses = []
        for batch in range(n_batches):
            train_images = next(iter(train_dataloader)).to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = model(train_images)
            loss = loss_fn(outputs[0], train_images)

            # perform backpropagation
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        losses_train.append(sum(epoch_losses) / len(epoch_losses))

        # adjust learning rate
        scheduler.step()

        print('epoch ', epoch, ' total loss: ', losses_train[-1])

        torch.save(model.decoder.state_dict(), param_file)

    # plot content loss, style loss, and total loss
    plt.plot(losses_train, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(output_plot_path)

#image classification model
def main():
    data_path = "/content/drive/MyDrive/ELEC475Lab3/"

    image_size = 512

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', type=str, default=data_path + "Data/COCO/COCO1k/", help='test directory')
    parser.add_argument('-e', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('-b', type=int, default=10, help='Batch size for training')
    parser.add_argument('-l', type=str, default=data_path + "encoder.pth", help='encoder weight file')
    parser.add_argument('-s', type=str, default=data_path + "decoder.pth", help='decoder weight file')
    parser.add_argument('-p', type=str, default=data_path + "loss_plot.png", help='loss plot file')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    opt = parser.parse_args()

    # load the encoder
    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(opt.l))

    my_transformations = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor()
    ])

    train_dataset = cd.custom_dataset(opt.content_dir, transform=my_transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.b, shuffle=True)



    # load model and other inputs to training function
    model = net.AdaIN_net(encoder=encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if opt.cuda == "y" or opt.cuda == "Y" else "cpu")
    n_epochs = int(opt.e)
    batch_size = int(opt.b)
    n_batches = int(len(train_dataset) / batch_size)
    output_plot_path = opt.p
    param_file = opt.s

    # train the model
    train(
        n_epochs,
        n_batches,
        optimizer,
        model,
        train_loader,
        loss_fn,
        scheduler,
        device,
        output_plot_path,
        param_file
    )


if __name__ == '__main__':
    main()


