import torch
import torchaudio
from torch.utils.data import DataLoader

from Dataset import GuitarStringDataset
from Losses import PretrainLoss
from Model import DDSPEncoderDecoderModel


def get_pretrain_loss(length, pluckposition, filter_params, criterion):
    loss = criterion(length, pluckposition, filter_params)

    return loss


def train(model, trainloader, pretrain_criterion, train_criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs in trainloader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            length, pluckposition, filter_params = model(inputs, strategy='encoder')
            loss = get_pretrain_loss(length, pluckposition, filter_params, pretrain_criterion)
            print(f"pretrain loss: {loss.item()}")
            if loss.item() != 0.0:
                loss.backward()
                optimizer.step()
            else:
                left, right = model(length=length, pluckposition=pluckposition, filter_params=filter_params,
                                    strategy='decoder')
                loss = train_criterion(left, inputs.squeeze(0))
                print(f"train loss: {loss.item()}")
                loss.backward()
                optimizer.step()
        #
        #     running_loss += loss.item()
        #
        # epoch_loss = running_loss / len(trainloader.dataset)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')


if __name__ == "__main__":
    audio, sr = torchaudio.load("guitar.wav")
    traindataset = GuitarStringDataset(audio.unsqueeze(0))
    traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True)

    model = DDSPEncoderDecoderModel(batch_size=1, C=8, D=3, n_filter_params=10, signal_length=64000, trainable=True)
    pretrain_criterion = PretrainLoss(0.1, 4)
    train_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = 'cpu'

    train(model, traindataloader, pretrain_criterion, train_criterion, optimizer, 1000, device)
