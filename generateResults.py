import numpy as np
from iterative_bayes_update import *

# Generate a "season" of results
N = 10 # Number of teams

rounds = 3 # Number of times everyone plays everyone

matchups = rounds*(int(N**2)) # Total number of mathups

# Team records: W - L
records = np.zeros((N, 2))
records.astype(int)

# Generate "quality"
quality = range(N) # For now, quality will just be the teams index


for i in range(matchups):
    team_A = int(i / (N)) % (N)
    team_B = i % (N)
    # Skip if team_A is team_B since teams cant play themselves
    if team_A != team_B:
        # Lower indexed team will be team A, higher will be B
        score_A = quality[team_A]
        score_B = quality[team_B]
        if score_A < score_B:
            records[team_A][1] += 1 # Add 1 to loss column
            records[team_B][0] += 1 # Add 1 to win column

print(records)

x, y = generate_prior() # Generte the starting prior

# Plot the posteriors for each team
for i in range(N):
    post = update_prior(x, y, int(records[i][0]), int(records[i][1]))
    plt.subplot(N, 1, i+1)
    plt.plot(x, post)

 
plt.show()


