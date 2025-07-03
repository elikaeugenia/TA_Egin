import os
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datareader import ShopeeComment
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model_ringan import TextCNN
from model_sedang import TextCNN
from model_berat import TextCNN
import numpy as np
import random
import matplotlib.pyplot as plt
import wandb

# Model Factory Function 
def get_model(model_name, vocab_size, num_classes, dropout):
    if model_name == "ringan":
        return TextCNN(vocab_size=vocab_size, embed_dim=100, num_classes=num_classes, do=dropout)
    elif model_name == "sedang":
        return TextCNN(vocab_size=vocab_size, embed_dim=300, num_classes=num_classes, do=dropout)
    elif model_name == "berat":
        return TextCNN(vocab_size=vocab_size, embed_dim=512, num_classes=num_classes, do=dropout)
    else:
        raise ValueError(f"Model size '{model_name}' is not supported.")

# Optimizer Factory Function 
def get_optimizer(optimizer_name, model_params, lr):
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(model_params, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr)
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(model_params, lr=lr)
    elif optimizer_name == "adamax":
        return torch.optim.Adamax(model_params, lr=lr)
    elif optimizer_name == "asgd":
        return torch.optim.ASGD(model_params, lr=lr)
    elif optimizer_name == "nadam":
        return torch.optim.NAdam(model_params, lr=lr)
    elif optimizer_name == "radam":
        return torch.optim.RAdam(model_params, lr=lr)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model_params, lr=lr)
    elif optimizer_name == "rprop":
        return torch.optim.Rprop(model_params, lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

# Training Function
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), correct / total

# Evaluation Function 
def evaluate(model, data_loader, criterion, device):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(data_loader), correct / total

# Main Training 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_name", type=str, default="adam", help="Optimizer name")
    parser.add_argument("--augment_prob", type=float, default=1.0, help="Probability of applying augmentation")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the CNN")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum token length for input")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--model_name", type=str, default="sedang", choices=["ringan", "sedang", "berat"], help="Model complexity: ringan, sedang, berat")
    parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment")
    args = parser.parse_args()


    wandb.init(
        project="cnn_shopeecomment",
        name="exp_20250630_s0",
        mode="online"   
        # name=args.name,
        # config=vars(args)
    )

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MAX_LEN = args.max_len
    NUM_CLASSES = args.num_classes
    model_name = args.model_name
    

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = os.path.join(os.getcwd(), 'models')
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Direktori output dibuat di: {output_dir}")
    except Exception as e:
        print(f"Error membuat direktori: {e}")
        return
    
    train_dataset = ShopeeComment(
        file_path="dataset.xlsx",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="shopee_datareader_simple_folds.json",
        random_state=2025,
        split="train",
        fold=0, # harusnya fold 0-4, akurasinya nanti rata rata dari 0-4 
        augmentasi_file="augmentasi.json"
    )
    
    val_dataset = ShopeeComment(
        file_path="dataset.xlsx",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="shopee_datareader_simple_folds.json",
        normalization_file="normalization_dict.json", 
        random_state=2025,
        split="val",
        fold=0,
        augmentasi_file="augmentasi.json",
        typo_prob=0,         
        swap_prob=0,         
        delete_prob=0,      
        synonym_prob=0,      
        phrase_prob=0
    )

    # Get vocab size and create dataloaders 
    vocab_size = train_dataset.tokenizer.vocab_size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Print debug information 
    # print("\nDebug info:")
    # print(f"Device: {device}")
    # print(f"Vocab size: {vocab_size}")
    # print(f"Num classes: {NUM_CLASSES}")
    # print("Train labels range:", torch.unique(next(iter(train_loader))['labels']))
    # print("Val labels range:", torch.unique(next(iter(val_loader))['labels']))
     
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model(model_name, vocab_size, NUM_CLASSES, args.dropout).to(device)

    from torchinfo import summary
    print("\nModel Summary:")
    summary(model, input_size=(BATCH_SIZE, MAX_LEN), dtypes=[torch.long], col_names=["input_size", "output_size", "num_params", "params_percent"])

    # initialize the optimizer
    optimizer = get_optimizer(args.optimizer_name, model.parameters(), LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler._last_lr[0]
        })

        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(output_dir, 'best_model.pth') 
            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

if __name__ == '__main__':
    main()
