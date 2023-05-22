import pickle
import matplotplib.pyplot as plt

decays = ["exponential", "polynomial"]
orders = [8, 12]

# DO NOT CHANGE THIS UNLESS IT IS CHANGED IN ALL OF THE
# EXPERIMENT FILES
true_noise_std = 0.5
start_plot_count_diff = 0
end_plot_count_diff = 0
start_count = 3
end_count = 25


def generate_diagram(decay, order):
    """
    Generates a diagram from the noises and the decay. In all the
    examples, the noise std is 0.5. As a result, The variance is 0.25.
    """
    # read the noises.pkl from the folder
    with open("{}_order_{}_data/noises.pkl", "rb") as f:
        noises = pickle.load(f)
    with open("{}_order_{}_data/eigenvalues.pkl", "rb") as f:
        eigenvalues = pickle.load(f)

        # now generate a miktex graph with this information.
    plt.axvline(x=order, color="r", linestyle="--")
    plt.plot(
        list(
            range(
                start_count + start_plot_count_diff,
                end_count + end_plot_count_diff,
            )
        ),
        noises,
    )
    plt.plot(
        list(
            range(
                start_count + start_plot_count_diff,
                end_count + end_plot_count_diff,
            )
        ),
        [
            true_noise_std
            for _ in range(
                start_count + start_plot_count_diff,
                end_count + end_plot_count_diff,
            )
        ],
    )
    plt.xlabel("Tested Order")
    plt.ylabel("Estimated Noise Standard Deviation")
    plt.show()


for decay in decays:
    for order in orders:
        print(
            "Generating diagram for {} decay with order {}".format(
                decay, order
            )
        )
        generate_diagram(decay, order)
        print("Done!")
        print("")
