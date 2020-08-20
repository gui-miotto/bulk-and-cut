import os
import csv
import shutil
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

def generate_pareto_animation(working_dir):
    # Create output directory
    figures_dir = os.path.join(working_dir, "pareto")
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir)

    population = _load_csv(working_dir=working_dir)
    for i in range(1, len(population)):
        frame_path = os.path.join(figures_dir, str(i).rjust(4, "0") + ".png")
        _build_a_frame(
            sub_population=population[:i],
            frame_path=frame_path,
            )


def _build_a_frame(sub_population, frame_path):
    pareto_front, dominated_set = _get_pareto_front(population=sub_population)
    arrow = _get_arrow(population=sub_population)
    _render_a_frame(
        title=len(sub_population),
        pareto_front=pareto_front,
        dominated_set=dominated_set,
        arrow=arrow,
        frame_path=frame_path,
        )


def _get_arrow(population):
    parent_id = population[-1]["parent"]
    if parent_id is None:
        return None
    child_nbulk = population[-1]["n_bulks"]
    parent_nbulk = population[parent_id]["n_bulks"]
    arrow_type = "bulk" if child_nbulk > parent_nbulk else "cut"

    child_cost = _individual_cost(population=population)
    parent_cost = _individual_cost(population=population, indv_id=parent_id)
    coords = np.vstack((child_cost, parent_cost - child_cost))

    return arrow_type, coords


def _individual_cost(population, indv_id=-1):
    indv = population[indv_id]
    n_pars = int(indv["n_pars"])
    neg_acc = float(indv["neg_acc"])
    return np.array([n_pars, neg_acc])


def _get_pareto_front(population):
    num_of_pars = np.array([ind["n_pars"] for ind in population])[:,np.newaxis]
    neg_accuracy = np.array([ind["neg_acc"] for ind in population])[:,np.newaxis]

    worst_at_num_pars = np.less(num_of_pars, num_of_pars.T)
    worst_at_accuracy = np.less(neg_accuracy, neg_accuracy.T)
    worst_at_both = np.logical_and(worst_at_num_pars, worst_at_accuracy)
    domination = np.any(worst_at_both, axis=0)

    costs = np.hstack((num_of_pars, neg_accuracy))
    dominated_set = costs[domination]
    pareto_front = costs[np.logical_not(domination)]
    return pareto_front, dominated_set


def _load_csv(working_dir):
    query = os.path.join(working_dir, "*csv")
    csv_path = glob(query)[0]
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_content = []
        for row in reader:
            csv_content.append({
                "n_pars" : int(row["n_parameters"]),
                "neg_acc" : -float(row["accuracy"]),
                "parent" : int(row["parent_id"]),
                "n_bulks" : int(row["bulk_counter"]),
                "n_cuts" : int(row["cut_counter"]),
            })
    return csv_content


def _render_a_frame(title, pareto_front, dominated_set, arrow, frame_path):
    #TODO: check the style I used on the master project
    # Global figure settings:
    plt.clf()
    plt.xlim((0., 4E7))
    plt.ylim((-100., -30.))
    plt.title(title)

    # Dominated solutions:
    if len(dominated_set) > 0:
        plt.scatter(
            x=dominated_set[:,0],
            y=dominated_set[:,1],
            marker="o",
            s=10.,
            alpha=.6,
            facecolors="none",
            edgecolors="g",
            )

    # Pareto-optimal solutions:
    pareto_front = pareto_front[np.argsort(pareto_front[:,0])]
    plt.plot(
        pareto_front[:,0],
        pareto_front[:,1],
        marker="*",
        linestyle="--",
        )

    # Parento-to-child arrow:
    ar_type = arrow[0]
    ar_coords = arrow[1]
    plt.arrow(
        x=ar_coords[0,0],
        y=ar_coords[0,1],
        dx=ar_coords[1,0],
        dy=ar_coords[1,1],
        color="m" if ar_type == "bulk" else "c",
        #width=.5,
        #head_width=2.,
        )

    plt.savefig(frame_path)
