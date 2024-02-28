import copy
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.pyplot import figure
from scipy.stats import sem

plt.rcParams["axes.xmargin"] = 0


def get_sweeps(project):
    sweeps = []
    for sweep in project:
        sweeps.append((sweep.name, sweep.id))
    return sweeps[::-1]


def get_run_links(sweeps, project_name):
    runs = []
    for sweep_name, sweep_link in sweeps:
        sweep = wandb.Api().sweep(f"lucacorbucci/{project_name}/{sweep_link}")
        run_list_sweep = []
        for run in sweep.runs:
            run_list_sweep.append(run.id)
        runs.append((sweep_name, run_list_sweep[::-1]))
    return runs


def get_run_data(run_links_per_sweep, project_name):
    run_data = []
    for sweep_name, sweep in run_links_per_sweep:
        print("Downloading data for a sweep")
        tmp_run_data = []
        for run_link in sweep:
            run = wandb.Api().run(f"lucacorbucci/{project_name}/{run_link}")

            tmp_run_data.append(pd.DataFrame(run.scan_history()))
        run_data.append((sweep_name, tmp_run_data))
    return run_data


def download_data(project_name):
    # project_name = "Dutch_Baseline_1"
    # check if we don't want to download the data again
    if not os.path.exists(
        f"/mnt/disk1/home/lcorbucci/plots_data/data_{project_name}.pkl"
    ):
        # plot(project_name)
        project = wandb.Api().project(project_name).sweeps()
        sweeps = get_sweeps(project)
        run_links = get_run_links(sweeps, project_name)
        data = get_run_data(
            run_links,
            project_name,
        )
        with open(
            f"/mnt/disk1/home/lcorbucci/plots_data/data_{project_name}.pkl", "wb"
        ) as f:
            dill.dump(data, f)
    else:
        with open(
            f"/mnt/disk1/home/lcorbucci/plots_data/data_{project_name}.pkl", "rb"
        ) as f:
            data = dill.load(f)
    return data


data_baseline_5 = download_data("Income_Baseline_1")
data_baseline_8 = download_data("Income_Baseline_2")
project_name = "Income_01_epsilon_1"
data = download_data(project_name)
project_name = "Income_005_epsilon_1"
data = download_data(project_name)
project_name = "Income_0075_epsilon_1"
data = download_data(project_name)
project_name = "Income_01_epsilon_2"
data = download_data(project_name)
project_name = "Income_005_epsilon_2"
data = download_data(project_name)
project_name = "Income_0075_epsilon_2"
data = download_data(project_name)
