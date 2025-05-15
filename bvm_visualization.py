# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import matplotlib.animation as animation

def get_samples_gamma(alpha, beta, n):
    return gamma.rvs(a=alpha, scale=1/beta, size=n)

MAX_TOTAL = 2000
SAMPLES = get_samples_gamma(3, 2, n=MAX_TOTAL)

def plot_posterior(num_samples, ax=None):
    """
    Plot the Gamma posterior for sample size num_samples,
    centered around lambda_star, with a marker line.
    If ax is None, create a new figure/axes and show the plot.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    lambda_star = 2/3  # pseudo true value
    samples = SAMPLES[:num_samples]

    # posterior parameters
    posterior_alpha = 1 + num_samples
    posterior_beta = 1 + np.sum(samples)

    post_var = posterior_alpha / posterior_beta**2
    post_sd  = np.sqrt(post_var)

    x_min = lambda_star - 0.5 #max(0, lambda_star - 4 * post_sd)
    x_max = lambda_star + 0.5 #lambda_star + 4 * post_sd
    x = np.linspace(x_min, x_max, 10000)

    pdf = gamma.pdf(x, a=posterior_alpha, scale=1/posterior_beta)

    # clear and plot
    ax.clear()
    ax.plot(x, pdf, linewidth=2, label=r"Posterior $\lambda \mid \mathbf{X}$")
    ax.axvline(lambda_star, linestyle='--', linewidth=2, color="orange", label=r"$\lambda^*$")
    # ax.text(lambda_star, ax.get_ylim()[1]*0.9, r'$\lambda^*$', 
    #         horizontalalignment='right', verticalalignment='top')

    ax.set_xlabel(r'Value of $\lambda$')
    ax.set_ylabel('Density')
    ax.set_title(f'Bernstein–von Mises Visualization (n={num_samples})')
    ax.legend(loc="upper left")
    ax.set_xlim(x_min, x_max)

    if created_fig:
        plt.show()
    else:
        return ax

# %%
if __name__ == "__main__": 
    fig, ax = plt.subplots()
    def update(frame):
        plot_posterior(num_samples=frame*10, ax=ax)
        return ax,

    ani = animation.FuncAnimation(
        fig, update, frames=int(MAX_TOTAL/10), blit=False, repeat=False
    )

    ani.save('bvm_demo.gif', writer='pillow', fps=20)

# %%
