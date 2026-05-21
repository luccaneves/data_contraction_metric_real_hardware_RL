import argparse
import os
from datetime import datetime
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import HydraulicDataset
from mlp_lyap_mean_bb import BB_Lyap_Mean  # Assuming you have this file for the Lyapunov mean version



def get_device(preferred_gpu=1):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if preferred_gpu < num_gpus:
            return torch.device(f"cuda:{preferred_gpu}")
        else:
            print(f"Warning: Only {num_gpus} GPU(s) available. Falling back to cuda:0.")
            return torch.device("cuda:0")
    return torch.device("cpu")


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Loaded checkpoint from '{path}' at epoch {start_epoch}")
    return start_epoch


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


def train(model, optimizer, criterion, dataloader, writer, log_dir, num_epochs, start_epoch=0,scheduller = None):
    device = get_device()
    model.to(device)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            #outputs = model(inputs)
            
            loss  = criterion(outputs, labels)
            #loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}] → Loss: {avg_loss:.6f}")
        scheduller.step(avg_loss)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        if (epoch) % 10 == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optimizer, epoch, avg_loss, log_dir)


def main():
    parser = argparse.ArgumentParser(description="Train MLP with optional checkpoint resume")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pth checkpoint to resume from")
    args = parser.parse_args()

    device = get_device()

    dataset = HydraulicDataset(
        csv_file='C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/' \
        'experimentsdata_20k_normalized_horizon_100.csv'
        #csv_file='/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/raw_data/0_2025-05-19-senoidal-300A-0.5to3hz/0_2025-05-19-senoidal-300A-0.5to3hz_normalized_with_last_x_feature.csv'
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8192,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = BB_Lyap_Mean(
        hidden_size=64,  # Increased from 64
        ts = 0.01,
        json_file_path='C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_data_new_mult_env_normalizing_constants.json'
        #json_file_path='/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/raw_data/0_2025-05-19-senoidal-300A-0.5to3hz/0_2025-05-19-senoidal-300A-0.5to3hz_normalizing_constants.json'
    )

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,    # Reduce by half (gentler than 0.1, which is aggressive)
        patience=8    # Faster reaction but not too sensitive
    )


    log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir)

    start_epoch = 0
    
    if False:
        check = "/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/train_model/runs/2025-08-26_13-01-32/model_272.pth"
        start_epoch = load_checkpoint(check, model, optimizer)

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=dataloader,
        writer=writer,
        log_dir=log_dir,
        num_epochs=400,  
        start_epoch=start_epoch,
        scheduller=scheduler
    )

    writer.close()


if __name__ == "__main__":
    main()
