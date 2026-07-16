import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import lightning as L


token_to_id = {
    "what": 0,
    "is": 1,
    "statquest": 2,
    "awesome": 3,
    "<EOS>": 4,
}

id_to_token = {
    token_id: token
    for token, token_id in token_to_id.items()
}


inputs = torch.tensor([
    [
        token_to_id["what"],
        token_to_id["is"],
        token_to_id["statquest"],
        token_to_id["<EOS>"],
        token_to_id["awesome"],
    ],
    [
        token_to_id["statquest"],
        token_to_id["is"],
        token_to_id["what"],
        token_to_id["<EOS>"],
        token_to_id["awesome"],
    ],
])


labels = torch.tensor([
    [
        token_to_id["is"],
        token_to_id["statquest"],
        token_to_id["<EOS>"],
        token_to_id["awesome"],
        token_to_id["<EOS>"],
    ],
    [
        token_to_id["is"],
        token_to_id["what"],
        token_to_id["<EOS>"],
        token_to_id["awesome"],
        token_to_id["<EOS>"],
    ],
])


dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)


class PositionEncoding(nn.Module):

    def __init__(self, d_model=2, max_len=6):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            start=0,
            end=max_len,
            step=1,
        ).float().unsqueeze(1)

        embedding_index = torch.arange(
            start=0,
            end=d_model,
            step=2,
        ).float()

        div_term = 1 / torch.tensor(10000.0) ** (
            embedding_index / d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, word_embeddings):
        return (
            word_embeddings
            + self.pe[:word_embeddings.size(0), :]
        )


class Attention(nn.Module):

    def __init__(self, d_model=2):
        super().__init__()

        self.d_model = d_model

        self.W_q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )

        self.W_k = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )

        self.W_v = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )

        self.row_dim = 0
        self.col_dim = 1

    def forward(
        self,
        encodings_for_q,
        encodings_for_k,
        encodings_for_v,
        mask=None,
    ):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        similarities = torch.matmul(
            q,
            k.transpose(
                dim0=self.row_dim,
                dim1=self.col_dim,
            ),
        )

        scaled_similarities = (
            similarities
            / (k.size(self.col_dim) ** 0.5)
        )

        if mask is not None:
            scaled_similarities = (
                scaled_similarities.masked_fill(
                    mask=mask,
                    value=-1e9,
                )
            )

        attention_percentages = F.softmax(
            scaled_similarities,
            dim=self.col_dim,
        )

        attention_scores = torch.matmul(
            attention_percentages,
            v,
        )

        return attention_scores


class DecoderOnlyTransformer(L.LightningModule):

    def __init__(
        self,
        num_tokens=4,
        d_model=2,
        max_len=6,
    ):
        super().__init__()

        L.seed_everything(seed=42)

        self.we = nn.Embedding(
            num_embeddings=num_tokens,
            embedding_dim=d_model,
        )

        self.pe = PositionEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        self.self_attention = Attention(
            d_model=d_model,
        )

        self.fc_layer = nn.Linear(
            in_features=d_model,
            out_features=num_tokens,
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)

        position_encoded = self.pe(
            word_embeddings
        )

        mask = torch.tril(
            torch.ones(
                (
                    token_ids.size(dim=0),
                    token_ids.size(dim=0),
                ),
                device=self.device,
            )
        )

        mask = mask == 0

        self_attention_values = (
            self.self_attention(
                position_encoded,
                position_encoded,
                position_encoded,
                mask=mask,
            )
        )

        residual_connection_values = (
            position_encoded
            + self_attention_values
        )

        output = self.fc_layer(
            residual_connection_values
        )

        return output

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=0.1,
        )

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        input_tokens, target_tokens = batch

        output = self(
            input_tokens[0]
        )

        loss = self.loss(
            output,
            target_tokens[0],
        )

        return loss


def generate_tokens(
    model,
    prompt_tokens,
    max_length=6,
):
    model_input = torch.tensor([
        token_to_id[token]
        for token in prompt_tokens
    ])

    predicted_ids = []

    while model_input.size(dim=0) < max_length:
        predictions = model(model_input)

        predicted_id = torch.argmax(
            predictions[-1, :]
        ).item()

        predicted_ids.append(
            predicted_id
        )

        if predicted_id == token_to_id["<EOS>"]:
            break

        model_input = torch.cat([
            model_input,
            torch.tensor([predicted_id]),
        ])

    predicted_tokens = [
        id_to_token[token_id]
        for token_id in predicted_ids
    ]

    return predicted_tokens


model = DecoderOnlyTransformer(
    num_tokens=len(token_to_id),
    d_model=2,
    max_len=6,
)


print("Before training")

before_training = generate_tokens(
    model,
    [
        "what",
        "is",
        "statquest",
        "<EOS>",
    ],
)

for token in before_training:
    print(token)


trainer = L.Trainer(
    max_epochs=30,
)

trainer.fit(
    model,
    train_dataloaders=dataloader,
)


print("After training")

first_answer = generate_tokens(
    model,
    [
        "what",
        "is",
        "statquest",
        "<EOS>",
    ],
)

for token in first_answer:
    print(token)


second_answer = generate_tokens(
    model,
    [
        "statquest",
        "is",
        "what",
        "<EOS>",
    ],
)

for token in second_answer:
    print(token)
