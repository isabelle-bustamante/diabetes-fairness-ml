import matplotlib.pyplot as plt
import numpy as np



def plot_tpr_by_group(tpr_dict, title="TPR by Group"):
    groups = list(tpr_dict.keys())
    values = list(tpr_dict.values())

    plt.figure(figsize=(6,4))
    bars = plt.bar(groups, values, color="skyblue")
    add_value_labels(bars)

    plt.ylim(0,1.1)
    plt.ylabel("TPR")
    plt.title(title)
    plt.xticks(rotation=30)
    plt.show()




def plot_fnr_by_group(fnr_dict, title="FNR by Group"):
    groups = list(fnr_dict.keys())
    values = list(fnr_dict.values())

    plt.figure(figsize=(6,4))
    bars = plt.bar(groups, values, color="salmon")
    add_value_labels(bars)

    plt.ylim(0,1.1)
    plt.ylabel("FNR")
    plt.title(title)
    plt.xticks(rotation=30)
    plt.show()



def plot_precision_by_group(precision_dict, title="Precision by Group"):
    groups = list(precision_dict.keys())
    values = list(precision_dict.values())

    plt.figure(figsize=(6,4))
    bars = plt.bar(groups, values, color="lightgreen")
    add_value_labels(bars)

    plt.ylim(0,1.1)
    plt.ylabel("Precision")
    plt.title(title)
    plt.xticks(rotation=30)
    plt.show()






def plot_equal_opportunity(tpr_dict, title="Equal Opportunity"):
    groups = list(tpr_dict.keys())
    values = list(tpr_dict.values())

    eo_gap = max(values) - min(values)

    plt.figure(figsize=(6,4))
    plt.bar(groups, values, color="lightblue")
    plt.ylim(0,1)
    plt.ylabel("TPR")
    plt.title(f"{title} (EO Gap = {eo_gap:.3f})")
    plt.xticks(rotation=30)
    plt.show()



def plot_eo_comparison(lr_gap_gender, nn_gap_gender,
                       lr_gap_race, nn_gap_race,
                       title="Equal Opportunity Gap Comparison"):
    models = ["LR Gender", "NN Gender", "LR Race", "NN Race"]
    gaps = [lr_gap_gender, nn_gap_gender, lr_gap_race, nn_gap_race]

    plt.figure(figsize=(7,4))
    plt.bar(models, gaps)
    plt.ylabel("EO Gap")
    plt.title(title)
    plt.xticks(rotation=30)
    plt.show()
