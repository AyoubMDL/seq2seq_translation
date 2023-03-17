import torch
from model import build_model
from constants import *
from data import DataLoader
import os

def translate_sentence(
    sentence, 
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25
):
    
    model = build_model(len(de_vocab), len(en_vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else :
            tokens = [token for token in sentence]
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = de_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        with torch.no_grad():
            hidden, cell = model.encoder(tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            inputs.append(pred_token)
            if pred_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    
    translated_sentence = " ".join(tokens[1:-1])
    
    return tokens

if __name__ == "__main__":
    loader = DataLoader(DATA_DIR)
    os.system("clear")

    sentence = input("Enter a german sentence to translate : ")
    translated_sentence = translate_sentence(sentence,
                                             de_nlp,
                                             loader.en_vocab,
                                             loader.de_vocab,
                                             LOWER,
                                             sos_token,
                                             eos_token,
                                             device)
    
    print(f"Translated sentence : {translated_sentence}")
