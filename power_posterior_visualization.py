# %%
from bvm_visualization import get_samples_gamma
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

MAX_TOTAL = 2000
SAMPLES = get_samples_gamma(3, 2, n=MAX_TOTAL)
GAMMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.00]

COLOR_MAP = {
    0.01: "pink",
    0.05: "green",
    0.1: "red",
    0.20: "purple",
    0.50: "brown",
    1.00: "lightblue",
}

def plot_power_posteriors(num_samples, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    lambda_star = 2/3  # pseudo-true value
    samples = SAMPLES[:num_samples]

    x_min = lambda_star - 0.5
    x_max = lambda_star + 0.5
    x = np.linspace(x_min, x_max, 10000)

    ax.clear()
    for gamma in GAMMAS:
        # posterior parameters
        posterior_alpha = 1 + num_samples * gamma
        posterior_beta  = 1 + np.sum(samples) * gamma

        pdf = stats.gamma.pdf(x, a=posterior_alpha, scale=1/posterior_beta)
        color = COLOR_MAP[gamma]
        ax.plot(
            x, pdf, linewidth=2,
            color=color,
            label=rf"$\gamma={gamma}$ posterior" +  r" $\lambda \mid \mathbf{X}$"
        )

    ax.axvline(
        lambda_star, linestyle='--',
        linewidth=2, color="orange",
        label=r"$\lambda^*$"
    )
    ax.set_xlabel(r'Value of $\lambda$')
    ax.set_ylabel('Density')
    ax.set_title(f'Power Posteriors Visualization (n={num_samples})')
    ax.legend(loc="upper left")
    ax.set_xlim(x_min, x_max)

    if created_fig:
        plt.show()
    else:
        return ax

if __name__ == "__main__": 
    fig, ax = plt.subplots()
    def update(frame):
        return plot_power_posteriors(frame * 10, ax=ax),

    ani = animation.FuncAnimation(
        fig, update, frames=int(MAX_TOTAL/10),
        blit=False, repeat=False
    )
    ani.save('power_posteriors.gif', writer='pillow', fps=10)
# %%
