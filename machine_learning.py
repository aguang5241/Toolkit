#!/usr/bin/env python
# -*-coding:utf-8 -*-

import torch

class NN(torch.nn.Module):
    def __init__(self, units):
        super(NN, self).__init__()
        modules = []
        for i in range(len(units) - 1):
            modules.append(torch.nn.Linear(units[i], units[i+1]))
        self.linears = torch.nn.ModuleList(modules=modules)

    def forward(self, data):
        for _, l in enumerate(self.linears):
            # Here is your activation function
            data = torch.nn.ReLU(l(data))
        return data