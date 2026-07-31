import torch

from src.dataset import ShakespeareDataset
from src.model import GPT_mini
from src.trainer import Trainer
from src.generation import Generator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset
dataset = ShakespeareDataset(sequence_length=128)

# Model
model = GPT_mini(
    dim=512,
    num_head=8,
    num_layers=5,
    scale_factor=4,
    vocab_size=dataset.vocab_size,
    max_sequance=128
)
# loading the model
checkpoint = torch.load("gpt_mini.pth", map_location=DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

# Generator
generator = Generator(
    model=model,
    tokenizer=dataset,
    device=DEVICE
)

text = generator.generate(
    prompt="To be",
    max_tokens=300,
    temperature=0.8,
    top_k=50
)

print("\nGenerated Text:\n")
print(text)