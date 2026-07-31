import torch
import streamlit as st

from src.dataset import ShakespeareDataset
from src.generation import Generator
from src.model import GPT_mini


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="GPT Mini",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 GPT Mini From Scratch")
st.write("Generate Shakespeare-style text using a GPT model implemented completely from scratch.")

# -----------------------------
# Load Dataset
# -----------------------------
dataset = ShakespeareDataset(sequence_length=128)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = GPT_mini(
        dim=512,
        num_head=8,
        num_layers=5,
        scale_factor=4,
        vocab_size=dataset.vocab_size,
        max_sequance=128
    )

    checkpoint = torch.load(
        "gpt_trained_with_noise.pth",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

generator = Generator(
    model=model,
    tokenizer=dataset,
    device=DEVICE
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Generation Settings")

temperature = st.sidebar.slider(
    "Temperature",
    0.1,
    2.0,
    0.8,
    0.1
)

top_k = st.sidebar.slider(
    "Top K",
    1,
    100,
    50
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    20,
    500,
    200
)

# -----------------------------
# Prompt
# -----------------------------
prompt = st.text_area(
    "Prompt",
    value="To be",
    height=150
)

# -----------------------------
# Generate
# -----------------------------
if st.button("🚀 Generate"):

    with st.spinner("Generating..."):

        output = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    st.success("Done!")

    st.subheader("Generated Text")

    st.text_area(
        "",
        value=output,
        height=400
    )