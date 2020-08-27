import os
import csv
import shutil
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import PIL

def generate_pareto_animation(working_dir):
    # Create output directory
    figures_dir = os.path.join(working_dir, "pareto")
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir)

    population = _load_csv(working_dir=working_dir)
    pop_size = len(population)
    for i in range(pop_size + 1):
        print(f"Generating frame {i} of {pop_size}")
        frame_path = os.path.join(figures_dir, str(i).rjust(4, "0") + ".png")
        _build_a_frame(
            sub_population=population[:i],
            frame_path=frame_path,
            )
    print("Generating GIF")
    _generate_gif(figs_dir=figures_dir)

def _build_a_frame(sub_population, frame_path):
    if len(sub_population) > 0:
        pareto_front, dominated_set = _get_pareto_front(population=sub_population)
        arrow = _get_connector(population=sub_population)
        _render_a_frame(
            title=_get_title_string(sub_population),
            pareto_front=pareto_front,
            dominated_set=dominated_set,
            arrow=arrow,
            frame_path=frame_path,
            )
    else:
        _render_a_frame(title="", frame_path=frame_path)

def _get_title_string(sub_population):
    ind_id = len(sub_population)
    parent_id = sub_population[-1]["parent"]
    title = "Newcomer:" + str(ind_id).rjust(4, "0") + "\n"
    if parent_id != -1:
        title += "(an offspring of " + str(parent_id).rjust(4, "0") + ")"
    return title

def _get_connector(population):
    parent_id = population[-1]["parent"]
    if parent_id == -1:
        return None
    child_nbulk = population[-1]["n_bulks"]
    parent_nbulk = population[parent_id]["n_bulks"]
    arrow_type = "bulk" if child_nbulk > parent_nbulk else "cut"

    child_cost = _individual_cost(population=population)
    parent_cost = _individual_cost(population=population, indv_id=parent_id)
    coords = np.vstack((child_cost, parent_cost))

    return arrow_type, coords

def _individual_cost(population, indv_id=-1):
    indv = population[indv_id]
    n_pars = int(indv["n_pars"])
    neg_acc = float(indv["neg_acc"])
    return np.array([n_pars, neg_acc])


def _get_pareto_front(population):
    # TODO: This function is not perfect: In the rare case of where two identical
    # solutions occur and they are not dominated, none of them will be put in the front.
    # Fix this.

    num_of_pars = np.array([ind["n_pars"] for ind in population])[:,np.newaxis]
    neg_accuracy = np.array([ind["neg_acc"] for ind in population])[:,np.newaxis]
    costs = np.hstack((num_of_pars, neg_accuracy))
    not_eye = np.logical_not(np.eye(len(costs))) # False in the main diagonal, True elsew.

    worst_at_num_pars = np.less_equal(num_of_pars, num_of_pars.T)
    worst_at_accuracy = np.less_equal(neg_accuracy, neg_accuracy.T)
    worst_at_both = np.logical_and(worst_at_num_pars, worst_at_accuracy)
    worst_at_both = np.logical_and(worst_at_both, not_eye)  # excludes self-comparisons
    domination = np.any(worst_at_both, axis=0)

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


def _pareto_front_coords(pareto_front, xmax=1E10, ymax=0.):
    pareto_front = pareto_front[np.argsort(pareto_front[:,0])]

    pareto_coords = []
    for i in range(len(pareto_front) - 1):
        pareto_coords.append(pareto_front[i])
        y1 = pareto_front[i][1]
        x2 = pareto_front[i + 1][0]
        pareto_coords.append([x2, y1])
    pareto_coords.append(pareto_front[-1])

    dominated_area = list(pareto_coords)
    dominated_area.append([xmax, pareto_coords[-1][1]])
    dominated_area.append([xmax, ymax])
    dominated_area.append([pareto_coords[0][0], ymax])

    return np.array(pareto_coords), np.array(dominated_area)


def _render_a_frame(title, frame_path, pareto_front=None, dominated_set=None, arrow=None):
    baseline = np.array([
        [8.80949400e+06, -7.69414740e+01],
        [2.84320000e+04, -5.86384692e+01],
        ])
    difandre = np.array([
        [4.27571700e+06, -8.13530869e+01],
        [ 3.64660000e+04, -8.00280941e+01],
        ])
    other_nets = np.array([
        [11.69E6, -93.87],
        [25.56E6, -87.99],
        [44.55E6, -90.41],
        [61.10E6, -90.20],
    ])
    fig_h = 6.
    fig_w = fig_h * 16. / 9.  # widescreen aspect ratio (16:9)

    # Global figure settings:
    plt.clf()
    plt.style.use("seaborn")
    plt.figure(figsize=(fig_w,fig_h))
    plt.xlim((1E4, 1E8))
    plt.ylim((-100., -40.))
    plt.title(title, fontdict={"family" : "monospace"})
    plt.xscale("log")
    plt.xlabel("number of parameters")
    plt.ylabel("negative accuracy")

    #Baselines
    baseline_front, _ = _pareto_front_coords(baseline)
    difandre_front, _ = _pareto_front_coords(difandre)
    plt.scatter(x=baseline[:,0], y=baseline[:,1], s=40.)
    plt.scatter(x=difandre[:,0], y=difandre[:,1], s=40.)
    plt.plot(baseline_front[:,0], baseline_front[:,1], label="baseline")
    plt.plot(difandre_front[:,0], difandre_front[:,1], label="difandre")
    plt.scatter(x=other_nets[:,0], y=other_nets[:,1], marker=".", color="k", label="other nets")

    # Dominated solutions:
    if dominated_set is not None and len(dominated_set) > 0:
        plt.scatter(
            x=dominated_set[:,0],
            y=dominated_set[:,1],
            marker=".",
            s=60.,
            alpha=.8,
            edgecolors="tab:gray",
            )

    # Pareto-optimal solutions:
    p_col = "tab:red"
    if pareto_front is not None:
        pareto_coords, dominated_area = _pareto_front_coords(pareto_front)
        plt.scatter(x=pareto_front[:,0], y=pareto_front[:,1], marker="*", color=p_col)
        plt.plot(pareto_coords[:,0], pareto_coords[:,1], alpha=.5, color=p_col, label="bulk n' cut")
        plt.fill(dominated_area[:,0], dominated_area[:,1], alpha=.2, color=p_col)

    # Parento-to-child arrow:
    if arrow is not None:
        ar_type = arrow[0]
        ar_coords = arrow[1]
        plt.plot(
            ar_coords[:,0],
            ar_coords[:,1],
            color="m" if ar_type == "bulk" else "c",
            )

    plt.legend(loc="lower left")
    plt.savefig(frame_path)
    plt.close()


def _generate_gif(figs_dir):
    query = os.path.join(figs_dir, "*.png")
    fig_paths = sorted(glob(query))
    imgs = [PIL.Image.open(fpath) for fpath in fig_paths]
    gif_path = os.path.join(figs_dir, "animated_pareto_front.gif")
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], loop=0, duration=10.)
