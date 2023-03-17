import spacy
import torch

DATA_DIR = "data"
MAX_LENGTH = 1_000
LOWER = True
sos_token = "<sos>"
eos_token = "<eos>"
en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

batch_size = 128
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model_path = "model/seq_model.pt"
n_epochs = 5
clip = 1.0
teacher_forcing_ratio = 0.5
