import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Bayes' Theorem applied to Beta distribution

# configure style
mpl.rc('text', usetex=True)
mpl.rc('font', size=15)
sns.set_style("darkgrid")
sns.set_context("talk", rc={"figure.figsize": (12, 8)}, font_scale=1)
current_palette = sns.color_palette()

def plot_prior(alpha, beta, ax=None):
    x = np.linspace(0, 1, 1000) # x-axis sample points
    y = scipy.stats.beta.pdf(x, alpha, beta) # Compute y values at sample x's

    if not ax:
        fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(r"$\theta$", fontsize=15)
    ax.set_ylabel(r"$P(\theta)$", fontsize=15)
    ax.set_title("Prior: BetaPDF({},{})".format(alpha,beta));
    plt.show()

def plot_posterior(heads, tails, alpha, beta, ax=None):
    x = np.linspace(0, 1, 1000) # x-axis sample points
    y = scipy.stats.beta.pdf(x, heads+alpha, tails+beta) # Compute y sample x's
        
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(r"$\theta$", fontsize=15)
    ax.set_ylabel(r"$P(\theta|D)$", fontsize=15)
    ax.set_title("Posterior after {} heads, {} tails, \
                 Prior: BetaPDF({},{})".format(heads, tails, alpha, beta));
    plt.show()


# 
plot_posterior(heads=1, tails=12, alpha=2, beta=2)
