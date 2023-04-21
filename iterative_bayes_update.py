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

# Let theta be the odds that Team A beats an average team

# Liklihood functions - One for if the team wins, and one for if the team loses
def win_liklihood(theta):
    return theta 
def loss_liklihood(theta):
    return 1 - theta


# This should be the factor that makes our probability one
def compute_marginal(liklihood, prior):
    return scipy.integrate.simps(np.multiply(liklihood, prior))

# Update the prior 
def compute_posterior(liklihood, prior):
    return np.multiply(liklihood, prior) / compute_marginal(liklihood, prior)


# Check to make sure the posterior integrates to 1
def check_is_pdf(x, posterior):
    I = scipy.integrate.simps(np.multiply(x, posterior))
    if (I-1) < 1e-14:
        return True
    else:
        return False


y = compute_posterior(loss_liklihood(x), y)

# Plot the posterior
plt.plot(x, y)

# Save the figure
plt.savefig("one_update.png")