import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


tokens = ["Troll2", "is", "great", "Gymkata"]


def plot_embeddings(w1, w2, offsets=None):
    data = {
        "w1": w1,
        "w2": w2,
        "token": tokens
    }

    df = pd.DataFrame(data)

    sns.scatterplot(
        data=df,
        x="w1",
        y="w2"
    )

    if offsets is None:
        offsets = [(0, 0)] * len(tokens)

    for index, token in enumerate(tokens):
        x_offset, y_offset = offsets[index]

        plt.text(
            df.w1[index] + x_offset,
            df.w2[index] + y_offset,
            token,
            horizontalalignment="left",
            size="medium",
            weight="semibold"
        )

    plt.show()


inputs = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

labels = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]
])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)


class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):
        super().__init__()

        L.seed_everything(seed=42)

        self.input_to_hidden = nn.Linear(
            in_features=4,
            out_features=2,
            bias=False
        )

        self.hidden_to_output = nn.Linear(
            in_features=2,
            out_features=4,
            bias=False
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_tensor):
        hidden = self.input_to_hidden(input_tensor)

        output = self.hidden_to_output(hidden)

        return output

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=0.1
        )

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch

        output_i = self(input_i)

        return self.loss(
            output_i,
            label_i
        )


model = WordEmbeddingWithLinear()


print("Before optimization")

for name, parameter in model.named_parameters():
    print(
        name,
        torch.round(parameter.data, decimals=2)
    )


weights = model.input_to_hidden.weight.detach()

plot_embeddings(
    weights[0].numpy(),
    weights[1].numpy()
)


trainer = L.Trainer(max_epochs=100)

trainer.fit(
    model,
    train_dataloaders=dataloader
)


print("After optimization")

for name, parameter in model.named_parameters():
    print(name, parameter.data)


weights = model.input_to_hidden.weight.detach()

plot_embeddings(
    weights[0].numpy(),
    weights[1].numpy(),
    [
        (-0.2, -0.3),
        (0, 0),
        (0, 0),
        (-0.3, 0.2)
    ]
)


softmax = nn.Softmax(dim=1)

for input_tensor in inputs:
    prediction = softmax(
        model(input_tensor.unsqueeze(0))
    )

    print(
        torch.round(prediction, decimals=2)
    )


word_embeddings = nn.Embedding.from_pretrained(
    model.input_to_hidden.weight.detach().T
)


vocab = {
    "Troll2": 0,
    "is": 1,
    "great": 2,
    "Gymkata": 3
}


print(word_embeddings.weight)

print(
    word_embeddings(
        torch.tensor(vocab["Troll2"])
    )
)

print(
    word_embeddings(
        torch.tensor(vocab["Gymkata"])
    )
)
