import numpy as np
import scipy
import matplotlib.pyplot as plt

# Numerically performs one Baysian update

# Parameters for the Beta distribution:
alpha = 2 
beta = 2

# Compute Prior
x = np.linspace(0, 1, 1000) # x-axis sample points
y = scipy.stats.beta.pdf(x, alpha, beta) # Prior

# Let theta be the odds that Team A beats an average opponent
# Liklihood functions - One for if the team wins, and one for if the team loses
def win_liklihood(theta):
    return theta 

def loss_liklihood(theta):
    return 1 - theta

# This should be the factor that makes our probability one
def compute_marginal(xx, liklihood, prior):
    return scipy.integrate.simps(np.multiply(liklihood, prior), x = xx)

# Update the prior 
def compute_posterior(xx, liklihood, prior):
    return np.multiply(liklihood, prior) / compute_marginal(xx, liklihood, prior)


# Check to make sure the posterior integrates to 1
def check_is_pdf(xx, posterior):
    I = scipy.integrate.simps(posterior, x = xx)
    if abs(I-1) < 1e-15:
        return True
    else:
        return False

# One Baysian update step where win is a boolean 
# True if team won
# False if team lost
def compute_posterior_step(x, y, win):
    if win:
        return compute_posterior(x, win_liklihood(x), y)
    else:
        return compute_posterior(x, loss_liklihood(x), y)

# Compute the posterior iteratively if the team loses losses games and wins wins games
def update_prior(x, prior, wins, losses):
    posterior = prior
    for i in range(wins):
        posterior = compute_posterior_step(x, posterior, True)
    for i in range(losses):
        posterior = compute_posterior_step(x, posterior, False)
    return posterior

y = update_prior(x, y, 1, 10)

# Plot the posterior
plt.plot(x, y)

# Show plot
plt.show()
