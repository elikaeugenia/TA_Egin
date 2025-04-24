import pandas as pd
import numpy as np
import re
import os
import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.data.find('tokenizers/punkt')
nltk.data.find('corpora/stopwords')
nltk.download('punkt_tab')

INDONESIAN_STOPWORDS = set(stopwords.words('indonesian'))

class ShopeeComment(Dataset):
    def __init__(
        self,
        file_path="dataset.xlsx",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="shopee_datareader_simple_folds.json",
        random_state=2025,
        split="train",
        fold=0,
        
    ):
        
        self.file_path = file_path
        self.folds_file = folds_file
        self.random_state = random_state
        self.split = split
        self.fold = fold
        
        # Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load Dataset
        self.load_data()
        # print(f"DataFrame: {self.df}") -jalanin pertama
        
        # Setup 5-Fold Cross Validation
        # Bagian ini membuat self.folds_indices yang berisi train_indices dan val_indices (fold 0-4)
        self.setup_folds()
        
        # Untuk print seluruh folds 0-4
        # print(f"self.folds_indices: {self.folds_indices}") 
       
        # Setup Indices
        # Bagian ini mempersiapkan self.indices yang berisi data yang akan di training yang akan dipilih berdasarkan 'split' dan 'fold'
        self.setup_indices()
        # print(f"self.indices: {self.indices}") # Hanya print yang sudah dipisahkan misalnya hanya fold 0 untuk training
        
    def __len__(self):
        # Mengembalikan panjang data
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Hanya mengambil nomor indeks dari data yang akan diambil
        idx = self.indices[idx]
        
        # Mengambil data komentar dari rating
        komentar = str(self.df.iloc[idx]['comment'])
        rating = self.df.iloc[idx]['rating']
        
        # Melakukan Pre-Processing
        comment_processed = self.preprocess_text(komentar)
        
        # Tokenisasi
        encoding = self.tokenizer(
            comment_processed,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        data = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(rating-1, dtype=torch.long),
            'original_text': komentar,
            'processed_text': comment_processed,
            'original_rating': rating,
            'original_index': idx,
        }
        
        return data
    
    def preprocess_text(self, text):
        # CASEFOLDING : konversi ke huruf kecil
        text = text.lower()
        # CLEANSING : hapus url
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # CLEANSING : menghapus special karakter
        text = re.sub(r'[^\w\s]', '', text)
        # menghapus spasi berlebih
        text = re.sub(r'\s+', ' ', text)
        # Tokenisasi
        words = nltk.word_tokenize(text)
        
        # Normalisasi kata tidak baku
        normalized_words = {
            'gk': 'tidak', 'gak': 'tidak',
            'nggak': 'tidak', 'ga': 'tidak',
            'bgt': 'banget', 'blom': 'belum',
            'jg': 'juga', 'klian': 'kalian',
            'sya': 'saya', 'aku': 'saya',
            'smpai': 'sampai', 'trus': 'terus',
            'ny': 'nya', 'nytrus': 'nya terus',
            'dr': 'dari', 'dri': 'dari',
            'tgl': 'tanggal', 'sudh': 'sudah',
            'da': 'ada',
        }
        
        # Normalization : mengganti kata tidak baku dengan kata baku
        words = [normalized_words.get(word, word) for word in words]
        # STOPWORDS : menghapus kata yang tidak penting
        words = [word for word in words if word not in INDONESIAN_STOPWORDS]
        # Menggabungkan kembali kata-kata yang sudah di tokenisasi
        text = ' '.join(words)
        
        return text             
        
    def setup_indices(self):
        # Mempersiapkan indices untuk data yang akan di training
        fold_key = f"fold_{self.fold}"
        
        if self.split == "train" :
            self.indices = self.folds[fold_key]['train_indices']
        else:
            self.indices = self.folds[fold_key]['val_indices']
       
    def setup_folds(self):
        # Check if folds file exists
        if os.path.exists(self.folds_file):
            self.load_folds()
        else: # Create folds if file does not exist
            self.create_folds()
            self.save_folds()
    
    # Load folds from JSON file
    def load_folds(self): 
        with open(self.folds_file, 'r') as f:
            folds_data = json.load(f)
            
        self.folds_indices = folds_data['fold_indices']
        self.folds = self.folds_indices
        print(f"Menggunakan {folds_data['n_folds']} folds dengan {folds_data['n_samples']} samples") 

        # for fold_name, indices in self.folds.items():
        #    print(f"{fold_name}:")
        #    print(f"  train_indices = {indices['train_indices']}")
        #    print(f"  val_indices   = {indices['val_indices']}")
        
    # Create Stratified K-Folds
    def create_folds(self):
        print(f"Membuat 5-folds cross-validation dengan random state {self.random_state}") 
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
     
        fold_indices = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['rating'])):
            fold_indices[f"fold_{fold}"] = {
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist()
            }
        
        # Save fold indices to JSON file
        with open(self.folds_file, 'w') as f:
            json.dump({
                'fold_indices' : fold_indices, 
                'n_samples' : len(self.df),
                'n_folds': 5,
                'random_state': self.random_state
            }, f)
            
        self.folds = fold_indices
        
    def save_folds(self):
        with open(self.folds_file, 'w') as f:
            json.dump({
                'fold_indices' : self.folds,
                'n_samples' : len(self.df),
                'n_folds': 5,
                'random_state': self.random_state
            }, f)
    
    def load_data(self):
        self.df = pd.read_excel(self.file_path) # Load the dataset excel
        self.df.columns = ['userName', 'rating', 'timestamp', 'comment'] # Rename columns
        self.df = self.df.dropna(subset=['comment', 'rating']) # Drop rows with NaN in 'comment' and 'rating' column
        self.df['rating'] = self.df['rating'].astype(int) # Convert rating to int
        self.df = self.df[(self.df['rating'] >= 1) & (self.df['rating'] <= 5)] # Filter rating between 1 and 5

if __name__ == "__main__":
    dataset = ShopeeComment(fold=1, split="val") # Intansi kelas 
    data = dataset[30] # Ambil data pertama
    print(f"Input IDs: {data['input_ids']}")
    print(f"Original Text: {data['original_text']}")
    print(f"Processed Text: {data['processed_text']}")
    print(f"Original Index: {data['original_index']}")
    
    
    # print(f"Index: {idx}, Comment: {comment}, Rating: {rating}") # Print data pertama
    # print(f"Processed Comment: {comment_processed}") # Print data yang sudah di pre-process