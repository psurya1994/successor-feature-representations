```
storage/01-avdsr.weights
storage/01-loss.p
Images of the results are stores as config1-*
test1.py has been used to generate these.
Contains weights and training curves for avdsr, 4rooms, lr:2e-3, iters:3e5, eps:1

storage/02-avdsr.weights
storage/02-loss.p
test2.py has been used to generate these.
Contains weights and training curves for avdsr, 4rooms, lr:2e-3, iters:3e5, eps:0.9, goalsDQN = [21, 28, 84, 91]

Using tmp.py to generate the visualization for these nets.

storage/03-rewards-0.9eps.p
Learning curves for test2.py, test3.py has been used to generate this.
```

To do:
- Keep goals in the same room, might give better performance for APSF
- Regenerate plots for option 1 and option 2 (least priority)
- P OOD, R chaning
	- Save network weights for avdsr option 1, init with option 2.
	- Save network weights for avdsr option 1. 
	- Use these to train various networks and plot the performance

