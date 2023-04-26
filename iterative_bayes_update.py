import numpy as np
import scipy
import matplotlib.pyplot as plt

# Numerically performs one Baysian update

def generate_prior():
    # Parameters for the Beta distribution:
    alpha = 2 
    beta = 2

    # Compute Prior
    x = np.linspace(0, 1, 1000) # x-axis sample points
    y = scipy.stats.beta.pdf(x, alpha, beta) # Initial prior
    return x, y

# Let theta be the odds that Team A beats an average opponent
# Liklihood functions - One for if the team wins, and one for if the team loses
def win_liklihood(theta):
    return theta 
def loss_liklihood(theta):
    return 1 - theta

# Compute denominator that makes the resulting prior integrate to 1
def compute_marginal(xx, liklihood, prior):
    return scipy.integrate.simps(np.multiply(liklihood, prior), x = xx)

# Update the prior with Bayes 
def compute_posterior(xx, liklihood, prior):
    return np.multiply(liklihood, prior) / compute_marginal(xx, liklihood, prior)


# Check to make sure the posterior integrates to (appoximatley) 1
def check_is_pdf(xx, posterior):
    I = scipy.integrate.simps(posterior, x = xx)
    if abs(I-1) < 1e-15:
        return True
    else:
        return False

# One Baysian update step where win is a boolean \
# win:
#   True if team won
#   False if team lost
def compute_posterior_step(x, y, win):
    if win:
        return compute_posterior(x, win_liklihood(x), y)
    else:
        return compute_posterior(x, loss_liklihood(x), y)

# Compute the posterior iteratively if the team loses l games and wins w games
def update_prior(x, prior, w, l):
    posterior = prior
    for i in range(w):
        posterior = compute_posterior_step(x, posterior, True)
    for i in range(l):
        posterior = compute_posterior_step(x, posterior, False)
    return posterior


# x, y = generate_prior()

# # Compute posteior distribution after a team wins 1 game and loses 10
# y = update_prior(x, y, 1, 10)

# # Plot the posterior
# plt.plot(x, y)
# plt.xlabel(r"$\theta$", fontsize=15)
# plt.ylabel(r"$P(\theta)$", fontsize=15)
# plt.title("Distribution after {} win(s) and {} lose(s)".format(1,10));

# # Show plot
# plt.show()
