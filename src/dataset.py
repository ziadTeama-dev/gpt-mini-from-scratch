import torch
from torch.utils.data import Dataset
import requests


class ShakespeareDataset(Dataset):
    def __init__(self, sequence_length=128):

        # Download Shakespeare text
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        text = requests.get(url).text

        self.sequence_length = sequence_length

        # create vocabulary
        self.chars = sorted(list(set(text)))

        self.vocab_size = len(self.chars)

        self.char_to_idx = {
            ch:i for i,ch in enumerate(self.chars)
        }

        self.idx_to_char = {
            i:ch for i,ch in enumerate(self.chars)
        }


        # encode text
        self.data = torch.tensor(
            [
                self.char_to_idx[c]
                for c in text
            ],
            dtype=torch.long
        )


    def __len__(self):
        return len(self.data) - self.sequence_length


    def __getitem__(self,idx):

        chunk = self.data[
            idx : idx + self.sequence_length + 1
        ]

        x = chunk[:-1]

        y = chunk[1:]

        return x, y



    def decode(self, tokens):

        return "".join(
            [
                self.idx_to_char[int(i)]
                for i in tokens
            ]
        )


    def encode(self,text):

        return torch.tensor(
            [
                self.char_to_idx[c]
                for c in text
            ],
            dtype=torch.long
        )