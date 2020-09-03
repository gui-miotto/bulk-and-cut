import matplotlib.pyplot as plt

fig_h = 6.
fig_w = fig_h * 16. / 9.  # widescreen aspect ratio (16:9)


def plot_learning_curves(
    ind_id:int,
    n_pars:int,
    curves:dict,
    fig_path:str,
    parent_loss:float = None,
    parent_accuracy:float = None,
    ):
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(fig_w, fig_h))
    fig.suptitle(f"Model {ind_id} \n {n_pars:,d} parameters")

    epochs = list(range(len(curves["validation_loss"])))

    color1 = "tab:red"
    ax1.set_ylabel("loss")
    ax1.plot(epochs, curves["validation_loss"], label="valid", color=color1)
    ax1.plot(epochs[1:], curves["train_loss"], label="train", color="tab:orange")
    #ax1.plot(curves["train_loss_at_eval"], label="train_at_eval", color="tab:pink")
    if parent_loss is not None:
        ax1.axhline(parent_loss, color=color1, linestyle="--")
    ax1.legend()

    color2 = "tab:blue"
    ax2.set_ylabel("accuracy (%)")
    ax2.set_xlabel("epoch")
    ax2.plot(epochs[1:], curves["validation_accuracy"], label="valid", color=color2)
    ax2.plot(epochs[1:], curves["train_accuracy"], label="train", color="b")
    if parent_accuracy is not None:
        ax2.axhline(parent_accuracy, color=color2, linestyle="--")
    ax2.legend()

    fig.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)