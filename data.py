import datasets
import torchtext
from utils import *
from constants import *


class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir

        dataset = datasets.load_dataset(
            "json",
            data_files={
                "train": self.data_dir + "/train.jsonl.txt",
                "valid": self.data_dir + "/val.jsonl.txt",
                "test": self.data_dir + "/test.jsonl.txt"
            }
        )

        train_data, valid_data, test_data = dataset["train"], dataset["valid"], dataset["test"]

        fn_kwargs = {
            "en_nlp": en_nlp, 
            "de_nlp": de_nlp, 
            "max_length": MAX_LENGTH,
            "lower": LOWER,
            "sos_token": sos_token,
            "eos_token": eos_token,
        }

        train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
        valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
        test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

        # Build english vocab
        self.en_vocab = torchtext.vocab.build_vocab_from_iterator(
            train_data["en_tokens"],
            min_freq=min_freq,
            specials=special_tokens,
        )

        # Build german vocab
        self.de_vocab = torchtext.vocab.build_vocab_from_iterator(
            train_data["de_tokens"],
            min_freq=min_freq,
            specials=special_tokens,  
        )

        assert self.en_vocab[unk_token] == self.de_vocab[unk_token]
        assert self.en_vocab[pad_token] == self.de_vocab[pad_token]

        unk_index = self.en_vocab[unk_token]
        self.pad_index = self.en_vocab[pad_token]

        # Using the set_default_index method we can set what value is returned 
        # when we try and get the index of a token outside of our vocabulary. 
        # In this case, the index of the unknown token 
        self.en_vocab.set_default_index(unk_index)
        self.de_vocab.set_default_index(unk_index)

        # Numericalize the data
        fn_kwargs = {
            "en_vocab": self.en_vocab,
            "de_vocab": self.de_vocab
        }

        train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
        valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
        test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

        # Transform "en_ids" and "de_ids" to tensors
        data_type = "torch"
        format_columns = ["en_ids", "de_ids"]

        self.train_data = train_data.with_format(
            type=data_type,
            columns=format_columns,
            output_all_columns=True
        )

        self.valid_data = valid_data.with_format(
            type=data_type, 
            columns=format_columns, 
            output_all_columns=True,
        )

        self.test_data = test_data.with_format(
            type=data_type, 
            columns=format_columns, 
            output_all_columns=True,
        )

    def build_loaders(self):
        train_data_loader = get_data_loader(self.train_data, batch_size, self.pad_index, shuffle=True)
        valid_data_loader = get_data_loader(self.valid_data, batch_size, self.pad_index)
        test_data_loader = get_data_loader(self.test_data, batch_size, self.pad_index)

        return train_data_loader, valid_data_loader, test_data_loader