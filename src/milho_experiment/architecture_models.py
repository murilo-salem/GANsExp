"""Geradores leves usados na busca R2/V8+R2 -> R5.

O módulo contém apenas arquiteturas. Persistência de pesos é deliberadamente
ausente: o protocolo mantém os modelos em memória durante cada fold.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.GroupNorm:
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            _norm(outputs), nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            _norm(outputs), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetLight(nn.Module):
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        self.e1 = ConvBlock(input_channels, width)
        self.e2 = ConvBlock(width, width * 2)
        self.e3 = ConvBlock(width * 2, width * 4)
        self.mid = ConvBlock(width * 4, width * 8)
        self.u3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
        self.d3 = ConvBlock(width * 8, width * 4)
        self.u2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.d2 = ConvBlock(width * 4, width * 2)
        self.u1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.d1 = ConvBlock(width * 2, width)
        self.out = nn.Sequential(nn.Conv2d(width, 3, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        mid = self.mid(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(mid), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int):
        super().__init__()
        hidden = max(8, skip_channels // 2)
        self.skip = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.score = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, skip, gate):
        return skip * self.score(self.skip(skip) + self.gate(gate))


class AttentionUNet(nn.Module):
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        self.e1 = ConvBlock(input_channels, width)
        self.e2 = ConvBlock(width, width * 2)
        self.e3 = ConvBlock(width * 2, width * 4)
        self.mid = ConvBlock(width * 4, width * 8)
        self.u3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
        self.a3 = AttentionGate(width * 4, width * 4)
        self.d3 = ConvBlock(width * 8, width * 4)
        self.u2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.a2 = AttentionGate(width * 2, width * 2)
        self.d2 = ConvBlock(width * 4, width * 2)
        self.u1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.a1 = AttentionGate(width, width)
        self.d1 = ConvBlock(width * 2, width)
        self.out = nn.Sequential(nn.Conv2d(width, 3, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        mid = self.mid(F.avg_pool2d(e3, 2))
        u3 = self.u3(mid)
        d3 = self.d3(torch.cat([u3, self.a3(e3, u3)], 1))
        u2 = self.u2(d3)
        d2 = self.d2(torch.cat([u2, self.a2(e2, u2)], 1))
        u1 = self.u1(d2)
        d1 = self.d1(torch.cat([u1, self.a1(e1, u1)], 1))
        return self.out(d1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3, bias=False),
            _norm(channels), nn.ReLU(inplace=True), nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False), _norm(channels),
        )

    def forward(self, x):
        return x + self.net(x)


class ResNet9(nn.Module):
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3), nn.Conv2d(input_channels, width, 7, bias=False),
            _norm(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            _norm(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            _norm(width * 4), nn.ReLU(inplace=True),
        ]
        layers.extend(ResidualBlock(width * 4) for _ in range(9))
        layers.extend([
            nn.ConvTranspose2d(width * 4, width * 2, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            _norm(width * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(width * 2, width, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            _norm(width), nn.ReLU(inplace=True), nn.ReflectionPad2d(3),
            nn.Conv2d(width, 3, 7), nn.Tanh(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(inputs + hidden, hidden * 4, 3, padding=1)

    def forward(self, x, state=None):
        if state is None:
            shape = (x.shape[0], self.hidden, x.shape[2], x.shape[3])
            h = x.new_zeros(shape)
            c = x.new_zeros(shape)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMLite(nn.Module):
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        if input_channels % 3:
            raise ValueError("ConvLSTM requer blocos temporais RRENIR de três canais")
        self.steps = input_channels // 3
        self.encoder = nn.Sequential(ConvBlock(3, width), nn.AvgPool2d(2),
                                     ConvBlock(width, width), nn.AvgPool2d(2))
        self.cell = ConvLSTMCell(width, width)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width // 2, 3, 3, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        state = None
        for frame in x.chunk(self.steps, dim=1):
            state = self.cell(self.encoder(frame), state)
        return self.decoder(state[0])


class SimVPLite(nn.Module):
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        if input_channels % 3:
            raise ValueError("SimVP requer blocos temporais RRENIR de três canais")
        self.steps = input_channels // 3
        self.encoder = nn.Sequential(ConvBlock(3, width), nn.AvgPool2d(2),
                                     ConvBlock(width, width), nn.AvgPool2d(2))
        self.mixer = nn.Sequential(
            ConvBlock(width * self.steps, width * 2),
            nn.Conv2d(width * 2, width, 1), nn.SiLU(),
            ResidualBlock(width), ResidualBlock(width),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width // 2, 3, 3, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        encoded = [self.encoder(frame) for frame in x.chunk(self.steps, dim=1)]
        return self.decoder(self.mixer(torch.cat(encoded, 1)))


class PatchDiscriminator(nn.Module):
    """PatchGAN que expõe ativações para feature matching."""
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        channels = (width, width * 2, width * 4)
        blocks = []
        previous = input_channels
        for index, current in enumerate(channels):
            layer = [nn.Conv2d(previous, current, 4, stride=2, padding=1)]
            if index:
                layer.append(_norm(current))
            layer.append(nn.LeakyReLU(0.2, inplace=True))
            blocks.append(nn.Sequential(*layer))
            previous = current
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(previous, 1, 3, padding=1)

    def forward(self, x, return_features: bool = False):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        score = self.out(x)
        return (score, features) if return_features else score


def make_generator(name: str, input_channels: int, width: int = 32) -> nn.Module:
    factories = {
        "unet_l1": UNetLight,
        "unet_light": UNetLight,
        "multiscale": UNetLight,
        "wgangp": UNetLight,
        "attention_unet": AttentionUNet,
        "resnet9": ResNet9,
        "convlstm_lite": ConvLSTMLite,
        "simvp_lite": SimVPLite,
    }
    if name not in factories:
        raise ValueError(f"gerador desconhecido: {name}")
    return factories[name](input_channels, width)


def make_discriminators(name: str, input_channels: int, width: int = 32) -> nn.ModuleList:
    count = 2 if name == "multiscale" else 1
    return nn.ModuleList(PatchDiscriminator(input_channels + 3, width) for _ in range(count))


def discriminator_scales(source, target, discriminators: Sequence[PatchDiscriminator],
                         return_features: bool = False):
    results = []
    pair = torch.cat([source, target], 1)
    for index, discriminator in enumerate(discriminators):
        if index:
            pair = F.avg_pool2d(pair, 2, count_include_pad=False)
        results.append(discriminator(pair, return_features=return_features))
    return results
