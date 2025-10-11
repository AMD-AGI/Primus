###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
from enum import Enum

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class Pipeline_Color(str, Enum):
    blue_1 = "#305496"
    blue_2 = "#B4C6E7"
    blue_3 = "royalblue"
    blue_4 = "lavender"
    green_1 = "#A9D08E"
    green_2 = "#375623"
    green_3 = "mintcream"
    green_4 = "seagreen"
    yellow_1 = "#d88c2b"
    yellow_2 = "#f6eab4"
    yellow_3 = "#d46014"
    yellow_4 = "#d44014"
    background = "#818181"


def get_color_dict(vp_size):
    color_dict = {
        "fwd_rect_color": [
            Pipeline_Color.blue_1,
            Pipeline_Color.blue_2,
            Pipeline_Color.blue_3,
            Pipeline_Color.blue_4,
        ],
        "bwd_rect_color": [
            Pipeline_Color.green_1,
            Pipeline_Color.green_2,
            Pipeline_Color.green_3,
            Pipeline_Color.green_4,
        ],
        "wgrad_rect_color": [
            Pipeline_Color.yellow_1,
            Pipeline_Color.yellow_2,
            Pipeline_Color.yellow_3,
            Pipeline_Color.yellow_4,
        ],
        "fwd_text_color": [
            "white",
            "black",
            "white",
            "black",
        ],
        "bwd_text_color": [
            "black",
            "white",
            "black",
            "white",
        ],
        "wgrad_text_color": [
            "gray",
            "black",
            "gray",
            "black",
        ],
    }

    for _ in range(4, vp_size):
        color_dict["fwd_rect_color"].append(
            np.random.rand(
                3,
            )
        )
        color_dict["bwd_rect_color"].append(
            np.random.rand(
                3,
            )
        )
        color_dict["wgrad_rect_color"].append(
            np.random.rand(
                3,
            )
        )
        color_dict["fwd_text_color"].append("white")
        color_dict["bwd_text_color"].append("white")
        color_dict["wgrad_text_color"].append("white")

    return color_dict


def get_bubble_ratio(data, iter_time, vp_size, num_mbs):
    non_bubble_time = 0
    for i in range(vp_size * num_mbs):
        non_bubble_time += data["fwd_end"][i] - data["fwd_start"][i]
        non_bubble_time += data["bwd_end"][i] - data["bwd_start"][i]
        if "wgrad_start" in data and len(data["wgrad_start"]) > 0:
            non_bubble_time += data["wgrad_end"][i] - data["wgrad_start"][i]

    bubble_ratio = (iter_time - non_bubble_time) / iter_time * 100
    return bubble_ratio


def get_task_data(task_list):
    task_data_list = []

    for task in task_list:
        task_data = {
            "title": task["title"],
            "iter_num": len(task["iter_to_vis"]),
            "iter_to_vis": task["iter_to_vis"],
        }

        config_path = os.path.join(task["log_path"], "config.json")
        with open(config_path) as f:
            task_data["config"] = json.load(f)

        pp_size = task_data["config"]["pp_size"]
        vp_size = task_data["config"]["vp_size"]
        num_mbs = task_data["config"]["num_mbs"]

        pp_rank_dict = {}
        for pp_rank in range(pp_size):
            file_path = os.path.join(task["log_path"], f"pp_rank_{pp_rank}.json")
            with open(file_path) as f:
                pp_rank_dict[pp_rank] = json.load(f)

        iter_dict = {}
        iter_time_max = 0
        for iter_idx in task_data["iter_to_vis"]:
            iter_dict[iter_idx] = {}
            iter_time = pp_rank_dict[0][str(iter_idx)]["total"]
            iter_dict[iter_idx]["iter_time"] = iter_time
            iter_time_max = max(iter_time, iter_time_max)
            for pp_rank in range(pp_size):
                iter_dict[iter_idx][pp_rank] = pp_rank_dict[pp_rank][str(iter_idx)]
                bubble_ratio = get_bubble_ratio(iter_dict[iter_idx][pp_rank], iter_time, vp_size, num_mbs)
                iter_dict[iter_idx][pp_rank]["bubble"] = bubble_ratio

        task_data["iters_dict"] = iter_dict
        task_data["iter_time_max"] = iter_time_max

        color_dict = get_color_dict(vp_size)
        task_data["color_dict"] = color_dict

        task_data_list.append(task_data)

    return task_data_list


def draw_rec(ax, text, x, y, w, h, text_color, rect_color, edge_color, font_size, ha, va):
    rect = patches.Rectangle((x, y), w, h, linewidth=0.4, edgecolor=edge_color, facecolor=rect_color)
    ax.add_patch(rect)
    rx, ry = rect.get_xy()
    cx = rx + rect.get_width() / 2.0
    cy = ry + rect.get_height() / 2.0
    ax.annotate(text, (cx, cy), color=text_color, fontsize=font_size, ha=ha, va=va)


def draw_pipeline(ax, iter_data, pp_size, vp_size, num_mbs, chunk_info_list, color_dict, draw_wgrad=False):
    series = ["fwd", "bwd"]
    if draw_wgrad:
        series += ["wgrad"]
    for pp_rank in range(pp_size):
        for serie in series:
            start = iter_data[pp_rank][serie + "_start"]
            end = iter_data[pp_rank][serie + "_end"]

            # modify chunk id and color by minibatch and chunk info
            minibatch_key = serie + "_minibatch"
            if minibatch_key in iter_data[pp_rank] and len(iter_data[pp_rank][minibatch_key]) > 0:
                for i in range(len(iter_data[pp_rank][minibatch_key])):
                    chunk_info_list[i]["id"] = iter_data[pp_rank][minibatch_key][i]

            chunk_key = serie + "_chunk"
            if chunk_key in iter_data[pp_rank] and len(iter_data[pp_rank][chunk_key]) > 0:
                for i in range(len(iter_data[pp_rank][chunk_key])):
                    chunk_id = iter_data[pp_rank][chunk_key][i]
                    chunk_info_list[i][serie + "_rect_color"] = color_dict[serie + "_rect_color"][chunk_id]
                    chunk_info_list[i][serie + "_text_color"] = color_dict[serie + "_text_color"][chunk_id]

            for i in range(vp_size * num_mbs):
                mbs_id = chunk_info_list[i]["id"]
                x = start[i]
                y = pp_size - pp_rank - 1
                w = end[i] - start[i]
                h = 1
                rect_color = chunk_info_list[i][serie + "_rect_color"]
                text_color = chunk_info_list[i][serie + "_text_color"]

                font_size = 3

                draw_rec(
                    ax, mbs_id, x, y, w, h, text_color, rect_color, "black", font_size, "center", "center"
                )

        bubble = iter_data[pp_rank]["bubble"]
        memory = iter_data[pp_rank]["memory"]
        text = f"(Bubble: {bubble:.2f}%)\n(Mem: {memory:.2f} GB)"

        draw_rec(ax, text, x + w, y, 10, h, "white", "none", "none", 5, "left", "center")


def get_chunk_info_list(num_mbs, pp_size, vp_size, color_dict):
    chunk_info_list = []
    id_list = [i + 1 for i in range(num_mbs)]
    color_idx_list = [0 for _ in range(num_mbs)]
    chunk_num = num_mbs // pp_size
    if vp_size > 1:
        color_idx_list = [[i] * pp_size for i in range(vp_size)] * chunk_num
        color_idx_list = sum(color_idx_list, [])
        id_list_split = [id_list[i : i + pp_size] for i in range(0, len(id_list), pp_size)]
        tmp_id_list = id_list_split.copy()
        for i in range(chunk_num, 0, -1):
            tmp_id_list.insert(i, id_list_split[i - 1] * (vp_size - 1))
        id_list = sum(tmp_id_list, [])
        id_list = sum(tmp_id_list, [])

    for i in range(num_mbs * vp_size):
        color_idx = color_idx_list[i]
        chunk_info_list.append(
            {
                "id": id_list[i],
                "fwd_rect_color": color_dict["fwd_rect_color"][color_idx],
                "bwd_rect_color": color_dict["bwd_rect_color"][color_idx],
                "wgrad_rect_color": color_dict["wgrad_rect_color"][color_idx],
                "fwd_text_color": color_dict["fwd_text_color"][color_idx],
                "bwd_text_color": color_dict["bwd_text_color"][color_idx],
                "wgrad_text_color": color_dict["wgrad_text_color"][color_idx],
            }
        )
    return chunk_info_list


def draw_sub(axs, task_data, iter_time_max):
    color_dict = task_data["color_dict"]
    pp_size = task_data["config"]["pp_size"]
    vp_size = task_data["config"]["vp_size"]
    num_mbs = task_data["config"]["num_mbs"]

    chunk_info_list = get_chunk_info_list(num_mbs, pp_size, vp_size, color_dict)

    for ax_idx, iter_idx in enumerate(task_data["iter_to_vis"]):
        ax = axs.flat[ax_idx] if type(axs) == np.ndarray else axs
        iter_data = task_data["iters_dict"][iter_idx]
        max_memory = max(iter_data[i]["memory"] for i in range(pp_size))
        iter_time = iter_data["iter_time"]
        sub_title = f"iteration {iter_idx} (iter_time: {iter_time:.0f}ms, max_mem: {max_memory:.2f}GB)"
        ax.set_title(sub_title, fontsize="small")
        ax.set_xlim((0, iter_time_max * 1.1))
        ax.set_ylim((0, pp_size))
        ax.set_xlabel("time(ms)", loc="right")
        ax.set_facecolor(Pipeline_Color.background)

        yaxis_list = [f"PP-{i}" for i in range(pp_size - 1, -1, -1)]
        y_pos = np.arange(len(yaxis_list)) + 0.5
        ax.set_yticks(y_pos, labels=yaxis_list)

        legend_list = []

        draw_wgrad = "wgrad_start" in iter_data[0] and len(iter_data[0]["wgrad_start"]) > 0

        for i in range(vp_size):
            fwd_rect = patches.Patch(
                color=color_dict["fwd_rect_color"][i], label="fwd" if vp_size == 1 else "fwd_vpp_" + str(i)
            )
            bwd_rect = patches.Patch(
                color=color_dict["bwd_rect_color"][i], label="bwd" if vp_size == 1 else "bwd_vpp_" + str(i)
            )
            legend_list.append(fwd_rect)
            legend_list.append(bwd_rect)
            if draw_wgrad:
                wgrad_rect = patches.Patch(
                    color=color_dict["wgrad_rect_color"][i],
                    label="wgrad" if vp_size == 1 else "wgrad_vpp_" + str(i),
                )
                legend_list.append(wgrad_rect)

        ax.legend(handles=legend_list, fontsize="6", loc="lower right")

        draw_pipeline(ax, iter_data, pp_size, vp_size, num_mbs, chunk_info_list, color_dict, draw_wgrad)


def draw(task_data_list):
    task_num = len(task_data_list)
    total_iter_num = sum(task_data_list[i]["iter_num"] for i in range(task_num))

    timeline_width = 12
    timeline_height = 3
    fig = plt.figure(
        constrained_layout=True, figsize=(timeline_width, total_iter_num * timeline_height), dpi=300
    )
    subfigs = fig.subfigures(task_num, 1)
    iter_time_max = max(task_data["iter_time_max"] for task_data in task_data_list)

    for task_idx in range(task_num):
        subfig = subfigs.flat[task_idx] if type(subfigs) == np.ndarray else subfigs
        task_data = task_data_list[task_idx]
        subfig.suptitle(task_data["title"], fontweight="bold")
        axs = subfig.subplots(task_data["iter_num"], 1)
        draw_sub(axs, task_data, iter_time_max)

    plt.show()


def main():
    show_exps = ["1f1b", "1f1b-interleaved", "zero-bubble-1f1b", "zbv", "v-half", "v-min"]
    task_list = [
        {
            "title": exp,
            "iter_to_vis": [i for i in range(5, 6)],
            "log_path": f"./pp_data/{exp}/",
        }
        for exp in show_exps
    ]
    matplotlib.use("WebAgg")

    task_data_list = get_task_data(task_list)
    draw(task_data_list)


if __name__ == "__main__":
    main()
