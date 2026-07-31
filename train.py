from src.dataset import ShakespeareDataset
from torch.utils.data import DataLoader
from src.model import GPT_mini
from src.trainer import Trainer


dataset = ShakespeareDataset(
    sequence_length=128
)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


model = GPT_mini(
    dim=512,
    num_head=8,
    num_layers=5,
    scale_factor=4,
    vocab_size=dataset.vocab_size,
    max_sequance=128
)


trainer = Trainer(
    model,
    loader,
    epochs=1,
    std_noise=.1,  #this using guassian noise as augmentation make it 0 if you don't want
    Logs_path="Logs/test2_with_noise",
    save_path="gpt_trained_with_noise.pth"
)


trainer.train()

