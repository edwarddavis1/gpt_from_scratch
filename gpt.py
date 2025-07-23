# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# %%
# hyperparameters
torch.manual_seed(123)

batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
eval_iters = 200
n_dim = 32
head_size = 32
# %%
# Load input text data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# %%
# ENCODE THE TOKENS

# Get all the unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encode based on char index
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}
encoder = lambda char_seq: [char_to_int[char] for char in char_seq]
decoder = lambda char_int_seq: "".join(
    [int_to_char[char_int] for char_int in char_int_seq]
)

# %%
# Process the data into train and test
data = torch.tensor(encoder(text))
n = len(data)

train_split = 0.9

train_data = data[: int(round(train_split * n))]
val_data = data[int(round(train_split * n)) :]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# %%
class attnHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.Q = nn.Linear(n_dim, head_size, bias=False)
        self.K = nn.Linear(n_dim, head_size, bias=False)
        self.V = nn.Linear(n_dim, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        attn_mat = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        attn_mat /= C**0.5

        # Causal masked attention + softmax
        attn_mat = attn_mat.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attn_mat = F.softmax(
            attn_mat, dim=-1
        )  # (B, T, T), dim=-1 normalises across rows

        logits = attn_mat @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return logits  # (B, T, C)


class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_dim)

        # Embedding to track the position of the token within the context window
        self.position_embedding = nn.Embedding(block_size, n_dim)
        self.lm_head = nn.Linear(n_dim, vocab_size)
        self.attn_head = attnHead(head_size=head_size)

    def forward(self, idx):
        B, T = idx.shape  # batch_size, context_length

        tok_emb = self.token_embedding(idx)  # batch_size, context_length, n_dim
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        emb = tok_emb + pos_emb
        emb = self.attn_head(emb)  # (B, T, n_dim)
        logits = self.lm_head(emb)  # batch_size, context_length, vocab_size  (B, T, C)

        return logits

    def generate(self, n_tokens):
        # Start from a zero token
        generation = torch.zeros(1, 1, dtype=torch.long, device=device)

        self.eval()
        with torch.no_grad():
            for _ in range(n_tokens):
                context = generation[:, -block_size:]  # batch, token, dim
                logits = self(context)

                # Select the logits for the last token
                last_logits = logits[:, -1, :]  # batch, dim

                # Softmax
                probs = F.softmax(last_logits, dim=-1)

                # Sample from softmax probs
                new_token = torch.multinomial(probs, num_samples=1)

                generation = torch.cat(
                    (generation, new_token), dim=1
                )  # batch, tokens + 1

        self.train()
        return generation


@torch.no_grad()
def estimate_loss(model, eval_iters=200):
    # Estimate the loss using a bunch of batches from both train and test
    model.eval()  # <- currently does nothing as no dropout or batch norm, but good practice

    out = {}
    for split in ["train", "test"]:
        losses = torch.zeros((eval_iters,))
        for i in range(eval_iters):
            x, y = get_batch(split)
            output = model(x)  # (B, T, C)
            output_flat = output.view(batch_size * block_size, vocab_size)
            y_flat = y.view(batch_size * block_size)
            loss = F.cross_entropy(output_flat, y_flat)
            losses[i] = loss

        out[split] = torch.mean(losses)

    model.train()
    return out


# %%
# INITIAL GENERATION

model = LanguageModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
epochs = 1000

print(decoder(model.generate(n_tokens=100)[0].tolist()))

# %%
train_losses = []
test_losses = []
epoch_checkpoints = 100
for epoch in range(epochs):

    if epoch % epoch_checkpoints == 0:
        losses = estimate_loss(model)
        print(f"Train loss: {losses['train']}\nTest loss: {losses['test']}")
        train_losses.append(losses["train"])
        test_losses.append(losses["test"])

    # Select batch
    # Put batch on device
    x, y = get_batch("train")
    output = model(x)

    # flatten out the batches
    output_flat = output.view(batch_size * block_size, vocab_size)
    y_flat = y.view(batch_size * block_size)

    loss = criterion(output_flat, y_flat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%
import numpy as np

plt.figure()
_ = plt.grid(alpha=0.2)
_ = plt.plot(train_losses, marker="o", label="Train")
_ = plt.plot(test_losses, marker="o", label="Test")
_ = plt.legend()
_ = plt.xlabel(f"Per {epoch_checkpoints} Epochs")
_ = plt.ylabel("Loss")


# %%
print(decoder(model.generate(n_tokens=100)[0].tolist()))


# %%
