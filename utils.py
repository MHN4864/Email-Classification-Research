import re
import pickle
import torch

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def encode_bilstm(text, vocab, max_len=200):
    words = text.split()[:max_len]
    indices = [vocab.get(w, vocab['<UNK>']) for w in words]
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    return torch.tensor(indices).unsqueeze(0)
