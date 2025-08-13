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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime

# Confusion Matrix Visualization Function
def plot_confusion_matrix(cm, model_name="TextCNN", class_names=['Negative', 'Positive'], 
                         precision=None, recall=None, f1=None, accuracy=None, 
                         save_path=None, show_plot=False):
    """
    Plot confusion matrix with style similar to the reference image
    
    Args:
        cm: confusion matrix from sklearn.metrics.confusion_matrix
        model_name: name of the model for the title
        class_names: list of class names
        precision, recall, f1, accuracy: metric scores to display
        save_path: path to save the plot (optional)
        show_plot: whether to display the plot
    """
    
    plt.figure(figsize=(8, 6))
    
    # Create the heatmap with blue colormap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': ''}, annot_kws={'size': 14, 'weight': 'bold'})
    
    # Set labels and title
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add metrics text below the plot if provided
    if any([precision, recall, f1, accuracy]):
        metrics_text = f"{model_name} Evaluation Metrics:\n"
        if accuracy is not None:
            metrics_text += f"Accuracy: {accuracy:.2f}\n"
        if precision is not None:
            metrics_text += f"Precision: {precision:.2f}\n"
        if recall is not None:
            metrics_text += f"Recall: {recall:.2f}\n"
        if f1 is not None:
            metrics_text += f"F1-Score: {f1:.2f}"
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   verticalalignment='bottom', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return plt.gcf()  # Return figure for wandb logging

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

# Enhanced Evaluation Function with Optional Plotting
def evaluate(model, data_loader, criterion, device, plot_cm=False, model_name="TextCNN", save_path=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix if requested
    fig = None
    if plot_cm:
        fig = plot_confusion_matrix(cm, model_name=model_name, 
                                   precision=precision, recall=recall, 
                                   f1=f1, accuracy=accuracy,
                                   save_path=save_path, show_plot=False)

    return total_loss / len(data_loader), accuracy, precision, recall, f1, cm, fig

# Main Training 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_name", type=str, default="adam", help="Optimizer name")
    parser.add_argument("--augment_prob", type=float, default=1.0, help="Probability of applying augmentation")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the CNN")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum token length for input")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--model_name", type=str, default="sedang", choices=["ringan", "sedang", "berat"], help="Model complexity: ringan, sedang, berat")
    parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment")
    args = parser.parse_args()
    
    # Initialize wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="cnn_shopeecomment_multifold",
        name=f"multifold_exp_{timestamp}",
        mode="online",
        config=vars(args)
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
    best_cm_fig = None
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        
        # Plot confusion matrix only on the last epoch or when we get best accuracy
        plot_cm = (epoch == EPOCHS - 1)
        cm_save_path = os.path.join(output_dir, f'confusion_matrix_epoch_{epoch+1}.png') if plot_cm else None
        
        val_loss, val_acc, val_precision, val_recall, val_f1, val_cm, cm_fig = evaluate(
            model, val_loader, criterion, device, 
            plot_cm=plot_cm, 
            model_name=f"TextCNN-{model_name}",
            save_path=cm_save_path
        )

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"Val   Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"Val   Confusion Matrix:\n{val_cm}")

        # Prepare wandb log
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "lr": scheduler._last_lr[0]
        }

        # Add confusion matrix to wandb if plotted
        if cm_fig is not None:
            log_dict["confusion_matrix"] = wandb.Image(cm_fig)
            plt.close(cm_fig)  # Close the figure to free memory

        wandb.log(log_dict)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(output_dir, 'best_model.pth') 
            torch.save(model.state_dict(), model_path)
            
            # Save confusion matrix for best model
            best_cm_save_path = os.path.join(output_dir, 'best_model_confusion_matrix.png')
            best_cm_fig = plot_confusion_matrix(val_cm, model_name=f"TextCNN-{model_name} (Best)", 
                                               precision=val_precision, recall=val_recall, 
                                               f1=val_f1, accuracy=val_acc,
                                               save_path=best_cm_save_path, show_plot=False)
            
            # Log best model to wandb
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(model_path)
            artifact.add_file(best_cm_save_path)
            wandb.log_artifact(artifact)
            
            # Also log the best confusion matrix
            wandb.log({"best_confusion_matrix": wandb.Image(best_cm_fig)})
            plt.close(best_cm_fig)

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.4f}")
    print(f"Best model and confusion matrix saved in: {output_dir}")
    
    wandb.finish()

if __name__ == '__main__':
    main() 