# Vision Transformer (ViT) – Mental Model

Study notes built from the MNIST ViT notebook (PyTorch): patch size 7, embedding 64, 4 heads, 4 encoder blocks.

## 1. The big idea

A ViT turns an image into a short 'sentence' of patches, then runs a normal Transformer on that sentence. A special \[CLS\] token reads the whole sentence and decides the class.

**Memory hook:** Cut the image (conv) → add a reporter (CLS) → give seat numbers (pos\_emb) → 4 × \[talk (attention) + think (MLP)\] → ask the reporter (x\[:, 0\]) → score 10 digits (Linear).

## 2. Architecture diagram

```
  Image  (B, 1, 28, 28)
      |
      v
+--------------------------------+
| PATCH EMBEDDING                |  Conv2d(1, 64, kernel=7, stride=7)
| cut into 4x4 = 16 patches      |  flatten + transpose
+--------------------------------+
      |   (B, 16, 64)
      v
  prepend [CLS] token             ->  (B, 17, 64)
      |
      v
  + position embedding            ->  (B, 17, 64)
      |
      v
+--------------------------------------------+
| TRANSFORMER ENCODER  x 4  (shape unchanged) |
|                                            |
|   x --> LayerNorm --> Multi-Head Attn --+  |
|   |                                     |  |
|   +----------------(+ residual)<--------+  |
|                                            |
|   x --> LayerNorm --> MLP 64-128-64 ----+  |
|   |                                     |  |
|   +----------------(+ residual)<--------+  |
+--------------------------------------------+
      |   (B, 17, 64)
      v
  take token 0 only: x[:, 0]      ->  (B, 64)
      |
      v
+--------------------------------+
| MLP HEAD: LayerNorm + Linear   |  Linear(64, 10)
+--------------------------------+
      |
      v
  10 class scores                 ->  (B, 10)
```

## 3. Shape journey (B = batch size)

| Step | Code | Shape |
| --- | --- | --- |
| Input image | MNIST | (B, 1, 28, 28) |
| Cut + embed | Conv2d(1, 64, kernel=7, stride=7) | (B, 64, 4, 4) |
| Flatten patches | flatten(2) | (B, 64, 16) |
| Swap axes | transpose(1, 2) | (B, 16, 64) |
| Add CLS | torch.cat((cls\_token, x), dim=1) | (B, 17, 64) |
| Add positions | x + pos\_emb | (B, 17, 64) |
| Encoder blocks x4 | TransformerEncoder | (B, 17, 64) |
| Read CLS | x\[:, 0\] | (B, 64) |
| Classify | LayerNorm + Linear(64, 10) | (B, 10) |

## 4. Each part, with the key code

### 4.1 Patch embedding – cut the image into 16 'words'

28x28 with 7x7 patches gives (28/7)^2 = 16 patches. Each patch has 49 pixels, and the conv turns those 49 numbers into a 64-number vector. Using kernel size = stride = patch size cuts non-overlapping patches and applies the same linear layer to each one.

```python
self.patch_emb = nn.Conv2d(num_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

def forward(self, x):
    x = self.patch_emb(x)      # (B, 64, 4, 4)
    x = x.flatten(2)           # (B, 64, 16)
    x = x.transpose(1, 2)      # (B, 16, 64)
    return x
```

### 4.2 CLS token and position embedding

The CLS token is one extra learnable vector at position 0. It starts empty and gathers a summary of all patches through attention. Attention has no sense of order, so the position embedding adds a learnable 'seat number' to every token.

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
self.pos_emb   = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim) * 0.02)

cls_token = self.cls_token.expand(x.shape[0], -1, -1)   # copy for every image in batch
x = torch.cat((cls_token, x), dim=1)                    # (B, 17, 64)
x = x + self.pos_emb
```

### 4.3 Transformer encoder block – talk, then think

Attention is communication: every token looks at all 17 tokens and pulls in useful information. The 4 heads each work in 16 dimensions (64 / 4) and look for different relationships. The MLP is individual thinking: each token is processed separately (64 → 128 → 64). Residual connections (x + ...) add notes without overwriting the original, which is what lets deep stacks train. LayerNorm keeps the numbers stable.

```python
self.multihead_attention = nn.MultiheadAttention(emb_dim, attention_head, batch_first=True)

def forward(self, x):
    residual1 = x
    x = self.layer_norm1(x)
    x = self.multihead_attention(x, x, x)[0]   # MIX: tokens talk
    x = x + residual1

    residual2 = x
    x = self.layer_norm2(x)
    x = self.mlp(x)                            # THINK: each token alone
    x = x + residual2
    return x
```

**Important:** batch\_first=True is required because the tensors are (B, tokens, dim). Without it PyTorch treats the first axis as the sequence and attention mixes across images instead of across patches.

### 4.4 MLP head – the decision

Only the CLS vector is used. It is normalized and mapped to 10 scores, one per digit.

```python
x = x[:, 0]                 # CLS token only: (B, 64)
x = self.mlp_head(x)        # LayerNorm + Linear(64, 10) -> (B, 10)
```

## 5. Training loop – the same 5 steps every batch

```python
optimizer.zero_grad()            # 1. clear old gradients
output = model(data)             # 2. forward pass: make guesses
loss = criterion(output, target) # 3. how wrong are the guesses?
loss.backward()                  # 4. which weights caused the error?
optimizer.step()                 # 5. nudge the weights
```

For evaluation, switch to model.eval() and wrap the loop in torch.no\_grad() so no gradients are tracked.

## 6. Train vs validation vs test

| Split | Purpose | How often used |
| --- | --- | --- |
| Training | Model learns from it (weights update) | Every epoch |
| Validation | Tune choices: learning rate, epochs, emb\_dim, blocks | Many times during development |
| Test | Honest final score on unseen data | Once, at the very end |

MNIST ships with 60,000 train and 10,000 test images. Loading with train=False gives the test set. For a proper setup, split the 60,000 into train and validation and keep the 10,000 as the true test set:

```python
from torch.utils.data import random_split
train_dataset, validation_data = random_split(full_train, [50000, 10000])
```

## 7. Numbers to remember

| Number | Meaning |
| --- | --- |
| 16 | Patches = (28 / 7)^2 |
| 17 | Tokens = 16 patches + 1 CLS |
| 64 | Width of every token (emb\_dim), constant through the network |
| 128 | MLP hidden size; only the inside of the MLP widens, then shrinks back to 64 |
| 4 x 16 | 4 heads, each working in 16 dims, giving 64 |
| 4 | Encoder blocks = 4 rounds of talk + think |

## 8. Class map and self-test

- PatchEmbedding: cut + embed
- vision\_transformer (inside its forward): CLS + position embedding, then chains the blocks
- TransformerEncoder: talk (attention) + think (MLP), repeated 4 times
- MLP\_head: read CLS, score 10 digits

**Self-test:** close the notes and redraw the shape table from memory. If you can explain why the shape stays (B, 17, 64) inside the encoder and why only x\[:, 0\] goes to the head, you understand the whole model.
