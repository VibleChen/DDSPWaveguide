from Core import get_reals_and_imgs, get_filter_params
from Losses import StabilityLoss


def get_pretrain_loss(params, penalty=0.1):
    nut_params, bridge_params, dispersion_params = get_filter_params(params)

    nut_a_reals, nut_a_imgs, nut_b_reals, nut_b_imgs = get_reals_and_imgs(nut_params)
    bridge_a_reals, bridge_a_imgs, bridge_b_reals, bridge_b_imgs = get_reals_and_imgs(bridge_params)
    dispersion_a_reals, dispersion_a_imgs, dispersion_b_reals, dispersion_b_imgs = get_reals_and_imgs(dispersion_params)

    stability_loss = StabilityLoss(penalty)

    loss = stability_loss(nut_a_reals, nut_a_imgs) + stability_loss(bridge_a_reals, bridge_a_imgs) + stability_loss(
        dispersion_a_reals, dispersion_a_imgs)

    return loss


def train(model, trainloader, criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs in trainloader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            latent = model(inputs, strategy='encoder')
            params = get_filter_params(latent)
            loss = get_pretrain_loss(params)

            if loss.item() != 0.0:
                loss.backward()
                optimizer.step()
            else:
                output = model(latent, strategy='decoder')
                loss = criterion(output, inputs)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
