import torch 
import tqdm
from constants import model_path

def train_fn(model, data_loader, optimizer,
             criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for batch in data_loader:
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss
    return epoch_loss / len(data_loader)

def train_model(n_epochs,
                model,
                train_data_loader,
                valid_data_loader,
                optimizer,
                criterion,
                clip,
                teacher_forcing_ratio,
                device
                ):
    best_valid_loss = float('inf')

    for _ in tqdm.tqdm(range(n_epochs)):
        train_loss = train_fn(
            model,
            train_data_loader,
            optimizer,
            criterion,
            clip,
            teacher_forcing_ratio,
            device
        )
        valid_loss = evaluate_fn(
            model,
            valid_data_loader,
            criterion,
            device
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
        
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')