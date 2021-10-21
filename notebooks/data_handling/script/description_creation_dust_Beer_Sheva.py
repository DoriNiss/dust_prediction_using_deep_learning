#!/usr/bin/env python
# coding: utf-8

import torch


description = {
    "idxs": [i for i in range(10)],
    "description": "Dust levels at Beer Sheva. dust_n is the PM10 levels of T + n hours, where T is the time of row (given in times file). delta_n is the difference dust_(n+6)-dust_n",
    "titles": [
        "Dust at T", # 0
        "Delta dust at T (dust @ T+6h)-(dust @ T)", # 1
        "Dust at T-24h", # 2
        "Delta dust at T-24h (dust @ T-24h)-(dust @ T-30h)", # 3
        "Dust at T+24h", # 4
        "Delta dust at T+24h (dust @ T+24h)-(dust @ T+18h)", # 5
        "Dust at T+48h", # 6
        "Delta dust at T+48h (dust @ T+48h)-(dust @ T+42h)", # 7
        "Dust at T+72h", # 8
        "Delta dust at T+72h (dust @ T+72h)-(dust @ T+66)", # 9
    ],
}


torch.save(description, "../../data/dust_description_pm10_BeerSheva_20000101_20210630_6h.pkl")




