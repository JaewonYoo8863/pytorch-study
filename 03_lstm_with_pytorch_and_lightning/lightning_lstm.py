import torch
import torch.nn as nn
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader


class LightningLSTM(L.LightningModule):

    def __init__(self):
        super().__init__()

        L.seed_everything(seed=42)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=1
        )

    def forward(self, input):

        input_transformed = input.view(
            len(input),
            1
        )

        lstm_output, _ = self.lstm(
            input_transformed
        )

        prediction = lstm_output[-1]

        return prediction

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=0.1
        )

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
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

model = LightningLSTM()

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
    max_epochs=300,
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
