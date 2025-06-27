import matplotlib.pyplot as plt
import numpy as np


def test_J2effect_exp_case2():
    """
    show the effect of J2 & Brownian noise by samples
    """
    exps = ["../../exp_case2/", "../../exp_case2_j2/"]
    colors = ["blue", "red"]
    labels = ["no J2","J2 + Brownian noise"]
    # set_publication_plot_style()
    T_PRIME_SPAN   = np.float32(np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20]))

    for t_prime in T_PRIME_SPAN:
        # print("[test] pdf(NN) marginalized to rphi at t=", t_prime)

        # Plotting the contour plot
        fig = plt.figure(figsize=(8, 6))

        for i in range(2):
            samples = np.load(exps[i]+"data/samples/samples_t{:.3f}.npy".format(t_prime))
            r_samples = samples[:,0]
            phi_samples = samples[:,2]
            # scatter samples of (r, phi) on to the plot
            plt.scatter(r_samples, phi_samples, s=30, c='white', linewidths=0.5, 
                        edgecolor=colors[i], alpha=1.0, label=labels[i])

        # Adding labels and title
        plt.xlabel(r"$r'$")
        plt.ylabel(r"$\phi'$")
        # plt.title(r"$p(r',\phi')$"+ "from NN and 200 Samples at t="+str(np.round(t_prime,2))+"T")
        plt.legend()
        plt.tight_layout(pad=0.2)
        # fig.savefig("figs/v2phat_idx3_nrphi"+str(np.round(t_prime,3))+".pdf", format='pdf')
        plt.show()


def main():
    test_J2effect_exp_case2()


if __name__ == "__main__":
    main()