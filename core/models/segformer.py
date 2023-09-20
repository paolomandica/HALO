import torch
import torch.nn.functional as F
import math
from torch import nn
from transformers import SegformerPreTrainedModel, SegformerConfig

from core.utils.hyperbolic import HyperMapper, HyperMLR


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config, num_classes=19, hyper=False, hfr=False):
        super().__init__(config)
        self.hyper = hyper
        self.hfr = hfr

        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        if hfr:
            self.hfr = nn.Sequential(
                nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size),
                nn.BatchNorm1d(config.decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size),
            )

        if hyper:
            self.mapper = HyperMapper()
            self.classifier = HyperMLR(config.decoder_hidden_size, num_classes)
        else:
            self.classifier = nn.Conv2d(
                config.decoder_hidden_size, num_classes, kernel_size=1
            )

        self.config = config
        self.post_init()

    def get_hfr_weights(self, x):
        feats = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        norm_weights = self.hfr(feats)
        norm_weights = norm_weights.view(-1, x.size(2) * x.size(3), x.size(1))
        norm_weights = torch.mean(norm_weights, dim=1, keepdim=False)
        norm_weights = norm_weights.view(-1, x.size(1), 1, 1)
        norm_weights = torch.clamp(norm_weights, min=1e-5)
        feats = x.reshape(-1, x.size(1), x.size(2) * x.size(3))
        feats = F.normalize(feats, dim=-1)
        feats = feats.reshape(-1, x.size(1), x.size(2), x.size(3))
        return feats, norm_weights

    def forward(
        self, encoder_hidden_states: torch.FloatTensor, size=None
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if (
                self.config.reshape_last_stage is False
                and encoder_hidden_state.ndim == 3
            ):
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.hfr:
            hidden_states, norm_weights = self.get_hfr_weights(hidden_states)
            hidden_states = hidden_states * norm_weights

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        if self.hyper:
            hidden_states = self.mapper.expmap(hidden_states, dim=1).double()
        logits = self.classifier(hidden_states).float()

        if size is not None:
            logits = F.interpolate(
                logits, size=size, mode="bilinear", align_corners=True
            )

        return logits, hidden_states


class SegFormerSegmentationHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        num_features: int = 4,
        hyper=False,
        hfr=False,
    ):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),  # why relu? Who knows
            nn.BatchNorm2d(channels),  # why batchnorm and not layer norm? Idk
        )
        self.hyper = hyper
        self.hfr = hfr
        if hfr:
            self.hfr = nn.Sequential(
                nn.Linear(channels, channels),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
            )
        if hyper:
            self.mapper = HyperMapper()
            self.hyper_mlr = HyperMLR(channels, num_classes)
        else:
            self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x, size=None):
        # x = torch.cat(features, dim=1)
        # x = self.fuse(x)
        x = x[0]
        feats = x.clone()
        if self.hfr:
            temp_out = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
            norm_weights = self.hfr(temp_out)
            norm_weights = norm_weights.view(-1, x.size(2) * x.size(3), x.size(1))
            norm_weights = torch.mean(norm_weights, dim=1, keepdim=False)
            norm_weights = norm_weights.view(-1, x.size(1), 1, 1)
            norm_weights = torch.clamp(norm_weights, min=1e-5)
            temp_out = x.reshape(-1, x.size(1), x.size(2) * x.size(3))
            temp_out = F.normalize(temp_out, dim=-1)
            temp_out = temp_out.reshape(-1, x.size(1), x.size(2), x.size(3))
            x = temp_out * norm_weights
        if self.hyper:
            feats = self.mapper.expmap(x, dim=1)
            x = self.hyper_mlr(feats.double()).float()
        else:
            x = self.predict(x)
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x, feats
