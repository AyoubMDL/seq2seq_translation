from data import DataLoader
from utils import *
from model import *
from constants import *
from train import train_model
import torch.optim as optim

def main():
    loader = DataLoader(DATA_DIR)
    train_data_loader, valid_data_loader, test_data_loader = loader.build_loaders()
    model = build_model(len(loader.de_vocab), len(loader.en_vocab))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=loader.pad_index)
    print("Training the model...")
    train_model(n_epochs, model, train_data_loader, valid_data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device)
    


if __name__ == "__main__":
    main()