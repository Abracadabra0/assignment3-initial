from transformers import AutoModel
import torch
from torch import nn


class ERNIEGuesser(nn.Module):
    def __init__(self, n_options=5, *,
                 n_layers,
                 n_hidden,
                 feature_concat,
                 use_ln=False,
                 dropout=0.):
        super().__init__()
        self.sentence_encoder = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.n_embed = 768
        self.n_options = n_options
        # fix parameters
        for name, param in self.sentence_encoder.named_parameters():
            param.requires_grad = False

        assert feature_concat in ['each', 'all'], "feature_concat should be either 'each' or 'all'"
        self.feature_concat = feature_concat
        if feature_concat == 'each':
            mlp = [nn.Linear(self.n_embed * 2, n_hidden)]
            if use_ln:
                mlp.append(nn.LayerNorm(n_hidden))
            mlp.append(nn.ReLU())
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                mlp.append(nn.Linear(n_hidden, n_hidden))
                if use_ln:
                    mlp.append(nn.LayerNorm(n_hidden))
                mlp.append(nn.ReLU())
                if dropout > 0:
                    mlp.append(nn.Dropout(dropout))
            mlp.append(nn.Linear(n_hidden, 1))
        else:
            mlp = [nn.Linear(self.n_embed * (n_options + 1), n_hidden)]
            if use_ln:
                mlp.append(nn.LayerNorm(n_hidden))
            mlp.append(nn.ReLU())
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            for _ in range(n_layers - 1):
                mlp.append(nn.Linear(n_hidden, n_hidden))
                if use_ln:
                    mlp.append(nn.LayerNorm(n_hidden))
                mlp.append(nn.ReLU())
                if dropout > 0:
                    mlp.append(nn.Dropout(dropout))
            mlp.append(nn.Linear(n_hidden, n_options))
        self.mlp = nn.Sequential(*mlp)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                riddle_input_ids,
                riddle_attention_mask,
                options_input_ids,
                options_attention_mask):
        """

        :param riddle_input_ids: (batch, length)
        :param riddle_attention_mask: (batch, length)
        :param options_input_ids: (batch, n_options, length)
        :param options_attention_mask: (batch, n_options, length)
        """
        batch_size = riddle_input_ids.shape[0]
        riddle = self.sentence_encoder(input_ids=riddle_input_ids,
                                       attention_mask=riddle_attention_mask).pooler_output
        # flatten options to fit into bert
        options_input_ids = options_input_ids.flatten(0, 1)
        options_attention_mask = options_attention_mask.flatten(0, 1)
        options = self.sentence_encoder(input_ids=options_input_ids,
                                        attention_mask=options_attention_mask).pooler_output
        options = options.reshape(batch_size, self.n_options, -1)
        if self.feature_concat == 'each':
            riddle = riddle[:, None, :]
            mlp_inp = torch.cat([riddle.repeat(1, self.n_options, 1), options], dim=-1)
            mlp_inp = mlp_inp.flatten(0, 1)  # (batch_size * n_options, length)
            mlp_out = self.mlp(mlp_inp)  # (batch_size * n_options, 1)
            mlp_out = mlp_out.reshape(batch_size, self.n_options)  # (batch_size, n_options)
        else:
            options = options.flatten(1, 2)  # (batch_size, length)
            mlp_inp = torch.cat([riddle, options], dim=-1)  # (batch_size, length)
            mlp_out = self.mlp(mlp_inp)  # (batch_size, n_options)
        return self.log_softmax(mlp_out)
