import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size * input_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        num_patches = seq_length // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size * input_dim)
        return self.linear(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = self._generate_positional_encoding(embed_dim, max_len)

    def _generate_positional_encoding(self, embed_dim, max_len):
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        idx = torch.arange(embed_dim, dtype=torch.float).unsqueeze(0)
        encoding = pos / 10000 ** (idx / embed_dim)
        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
        return encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)
        encoding = self.encoding[:, :x.size(1)].to(x.device)  # 将 self.encoding 转移到与 x 相同的设备
        return x + encoding


class PretrainModel(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim, num_layers, dropout, patch_size, device):
        super(PretrainModel, self).__init__()
        self.device = device
        self.patch_embedding = PatchEmbedding(feature_dim, hidden_dim, patch_size).to(device)
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len).to(device)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, batch_first=True),
            num_layers=num_layers
        ).to(device)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, batch_first=True),
            num_layers=num_layers
        ).to(device)
        self.output_layer = nn.Linear(hidden_dim, feature_dim).to(device)

    def forward(self, x, mask=None, is_pretraining=True):
        x = x.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        Tmask = ~mask

        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        batch_size, seq_length, feature_dim = x.size()

        encoder_output = self.encoder(
            x, src_key_padding_mask=Tmask.view(batch_size, -1) if Tmask is not None else None
        ) * mask.unsqueeze(-1)

        combined_output = encoder_output

        if is_pretraining:
            decoder_output = self.decoder(
                combined_output, src_key_padding_mask=Tmask.view(batch_size, -1) if Tmask is not None else None
            ) * mask.unsqueeze(-1)
            return self.output_layer(decoder_output.view(batch_size * seq_length, -1))
        else:
            return combined_output

