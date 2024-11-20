def train(model, data_loader, loss, optimizer, device):
    """

    :param model:
    :param data_loader:
    :param loss:
    :param optimizer:
    :param device:
    :return:
    """

    for bacth in data_loader:
        optimizer.zero_grad()
        enc_inputs, dec_inputs, dec_outputs = bacth
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        dec_outputs = dec_outputs.to(device)

        logits = model(enc_inputs, dec_inputs)
        loss_data = loss(logits, dec_outputs.view(-1))
        loss_data.backward()
        print(loss_data)
        optimizer.step()
