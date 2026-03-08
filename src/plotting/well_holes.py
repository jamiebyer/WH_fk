import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np


def ft_to_m(depths):
    return np.array(depths) * 0.3048


def well_hole_plotting(site):
    # Well hole data
    if site == "WH01":
        # 23 Yukon Wild Rd.
        well_holes = {
            "A": {
                "depth": [0, 20, 120, 128, 240],
                "unit": "ft",
                "primary_material": ["Clay", "Clay", "Gravel", "Bedrock"],
                "secondary_material": [None, None, "Clay", None],
            },
            "B": {
                "depth": [0, 26, 27, 125],
                "unit": "ft",
                "primary_material": ["Clay", None, "Clay"],
                "secondary_material": [None, None, None],
            },
            "C": {
                "depth": [0, 130, 450],
                "unit": "m",
                "primary_material": ["Sand", "Bedrock"],
                "secondary_material": ["Clay", None],
            },
            "D": {
                "depth": [0, 25, 45, 68, 69, 478],
                "unit": "ft",
                "primary_material": ["Clay", "Clay", "Clay", None, "Bedrock"],
                "secondary_material": [None, "Sand, Gravel", None, None, None],
            },
            "E": {
                "depth": [0, 6, 95, 114, 144, 147, 155],
                "unit": "ft",
                "primary_material": [None, "Clay", "Clay", "Sand", "Sand", "Clay"],
                "secondary_material": [
                    None,
                    None,
                    "Sand, Gravel",
                    "Clay, Silt",
                    "Gravel",
                    "Sand, Gravel",
                ],
            },
        }
    elif site == "WH02":
        well_holes = {
            "F": {
                "depth": [0, 12, 16, 54, 58, 178],
                "unit": "m",
                "primary_material": ["Gravel", "Sand", "Gravel", "Sand", "Gravel"],
                "secondary_material": [
                    "Sand",
                    "Gravel",
                    "Sand",
                    "Gravel",
                    "Sand",
                ],
            },
            "G": {
                "depth": [
                    0,
                    9,
                    43,
                    78,
                    87,
                    104,
                    113,
                    199.6,
                    202,
                    205,
                    261,
                    316,
                    328,
                    340,
                ],
                "unit": "ft",
                "primary_material": [
                    "Sand",
                    "Silt",
                    "Clay",
                    "Clay",
                    "Sand",
                    "Sand",
                    "Silt",
                    "Silt",
                    "Silt",
                    "Silt",
                    "Silt",
                    "Till",
                    "Bedrock",
                ],
                "secondary_material": [
                    None,
                    None,
                    None,
                    None,
                    "Gravel",
                    "Silt",
                    "Clay",
                    "Sand, Gravel",
                    "Gravel, Sand",
                    "Clay",
                    "Clay",
                    None,
                    None,
                ],
            },
        }

    color_dict = {
        "Clay": "saddlebrown",
        "Sand": "wheat",
        "Bedrock": "dimgrey",
        "Gravel": "lightgrey",
        "Till": "lightsteelblue",
        "Silt": "olive",
    }

    fig, ax = plt.subplots()

    # diff materials: Clay, Sand, Bedrock, Gravel, None
    for i, (k, v) in enumerate(well_holes.items()):
        # k is x label
        depths = v["depth"]
        # if unit is ft, need to convert to m
        if v["unit"] == "ft":
            depths = ft_to_m(depths)
        for j in range(len(depths) - 1):
            if v["primary_material"][j] is None:
                continue

            ax.add_patch(
                Rectangle(
                    (2 * i, depths[j]),
                    1.75,
                    depths[j + 1] - depths[j],
                    color=color_dict[v["primary_material"][j]],
                )
            )

            material_label = v["primary_material"][j]
            if v["secondary_material"][j] is not None:
                material_label += " (" + v["secondary_material"][j] + ")"
            ax.text(2 * i, depths[j] + 6, material_label)
    plt.xticks(np.arange(1, 2 * len(well_holes), 2), well_holes.keys())

    plt.xlim(0, 2 * len(well_holes))
    plt.ylim(0, 200)
    # plt.ylim(0, 175)
    # plt.ylim(0, 500)

    plt.gca().invert_yaxis()
    plt.ylabel("depth (m)")

    plt.title(site)

    plt.show()
