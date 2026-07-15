import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.uniform import Uniform
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


class WordEmbeddingFromScratch(L.LightningModule):

    def __init__(self):
        super().__init__()

        L.seed_everything(seed=42)

        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.input1_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.input2_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.input2_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.input3_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.input3_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.input4_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.input4_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.output1_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.output1_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.output2_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.output2_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.output3_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.output3_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.output4_w1 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )
        self.output4_w2 = nn.Parameter(
            Uniform(min_value, max_value).sample()
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_tensor):
        input_tensor = input_tensor[0]

        top_hidden = (
            input_tensor[0] * self.input1_w1
            + input_tensor[1] * self.input2_w1
            + input_tensor[2] * self.input3_w1
            + input_tensor[3] * self.input4_w1
        )

        bottom_hidden = (
            input_tensor[0] * self.input1_w2
            + input_tensor[1] * self.input2_w2
            + input_tensor[2] * self.input3_w2
            + input_tensor[3] * self.input4_w2
        )

        output1 = (
            top_hidden * self.output1_w1
            + bottom_hidden * self.output1_w2
        )

        output2 = (
            top_hidden * self.output2_w1
            + bottom_hidden * self.output2_w2
        )

        output3 = (
            top_hidden * self.output3_w1
            + bottom_hidden * self.output3_w2
        )

        output4 = (
            top_hidden * self.output4_w1
            + bottom_hidden * self.output4_w2
        )

        return torch.stack([
            output1,
            output2,
            output3,
            output4
        ])

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
            label_i[0]
        )


model = WordEmbeddingFromScratch()

print("Before optimization")

for name, parameter in model.named_parameters():
    print(
        name,
        torch.round(parameter.data, decimals=2)
    )


plot_embeddings(
    [
        model.input1_w1.item(),
        model.input2_w1.item(),
        model.input3_w1.item(),
        model.input4_w1.item()
    ],
    [
        model.input1_w2.item(),
        model.input2_w2.item(),
        model.input3_w2.item(),
        model.input4_w2.item()
    ]
)


trainer = L.Trainer(max_epochs=100)

trainer.fit(
    model,
    train_dataloaders=dataloader
)


print("After optimization")

for name, parameter in model.named_parameters():
    print(
        name,
        torch.round(parameter.data, decimals=2)
    )


plot_embeddings(
    [
        model.input1_w1.item(),
        model.input2_w1.item(),
        model.input3_w1.item(),
        model.input4_w1.item()
    ],
    [
        model.input1_w2.item(),
        model.input2_w2.item(),
        model.input3_w2.item(),
        model.input4_w2.item()
    ],
    [
        (-0.2, 0.1),
        (0, 0),
        (0, 0),
        (-0.3, -0.3)
    ]
)


softmax = nn.Softmax(dim=0)

for input_tensor in inputs:
    prediction = softmax(
        model(input_tensor.unsqueeze(0))
    )

    print(
        torch.round(prediction, decimals=2)
    )
