import torch
import torch.nn as nn
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader


class LSTMbyHand(L.LightningModule):

    def __init__(self):
        super().__init__()

        L.seed_everything(seed=42)

        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.blr1 = nn.Parameter(torch.tensor(0.0))

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.bpr1 = nn.Parameter(torch.tensor(0.0))

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.bp1 = nn.Parameter(torch.tensor(0.0))

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std))
        self.bo1 = nn.Parameter(torch.tensor(0.0))

    def lstm_unit(self, input_value, long_memory, short_memory):

        long_remember_percent = torch.sigmoid(
            short_memory * self.wlr1
            + input_value * self.wlr2
            + self.blr1
        )

        potential_remember_percent = torch.sigmoid(
            short_memory * self.wpr1
            + input_value * self.wpr2
            + self.bpr1
        )

        potential_memory = torch.tanh(
            short_memory * self.wp1
            + input_value * self.wp2
            + self.bp1
        )

        updated_long_memory = (
            long_memory * long_remember_percent
            + potential_remember_percent * potential_memory
        )

        output_percent = torch.sigmoid(
            short_memory * self.wo1
            + input_value * self.wo2
            + self.bo1
        )

        updated_short_memory = (
            torch.tanh(updated_long_memory) * output_percent
        )

        return updated_long_memory, updated_short_memory

    def forward(self, input):

        long_memory = 0
        short_memory = 0

        for input_value in input:
            long_memory, short_memory = self.lstm_unit(
                input_value,
                long_memory,
                short_memory
            )

        return short_memory

    def configure_optimizers(self):
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):

        input_i, label_i = batch
        output_i = self(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)

        if label_i.item() == 0:
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss.mean()


inputs = torch.tensor([
    [0.0, 0.5, 0.25, 1.0],
    [1.0, 0.5, 0.25, 1.0]
])

labels = torch.tensor([0.0, 1.0])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = LSTMbyHand()

print("Before optimization")

print(
    "Company A:",
    model(torch.tensor([0.0, 0.5, 0.25, 1.0])).detach()
)

print(
    "Company B:",
    model(torch.tensor([1.0, 0.5, 0.25, 1.0])).detach()
)

trainer = L.Trainer(
    max_epochs=5000,
    log_every_n_steps=2
)

trainer.fit(
    model,
    train_dataloaders=dataloader
)

print("After optimization")

print(
    "Company A:",
    model(torch.tensor([0.0, 0.5, 0.25, 1.0])).detach()
)

print(
    "Company B:",
    model(torch.tensor([1.0, 0.5, 0.25, 1.0])).detach()
)

for name, parameter in model.named_parameters():
    print(name, parameter.data)
