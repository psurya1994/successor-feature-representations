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

04
Same as 02, but with dying=0.1 instead of dying = 0.01 in four rooms. That might help us get better steady state sample distribution.

05
Running test3. Same as the relation between 02 and 03. Here between 04 and 05.


06, 07 same as above but with eps=0.8. Because eps=0.9 SR visualization isn't good. SR tapers off too quick.

08,09 for eps = 0.4, dying = 0.05 because the above tapers off too quick too! (09 eps has the wrong label in the saved file, be careful!)

10,11 -- use 1 DQN (goal: 28). everything else same as above. eps = 0.8, because 0.4 seemed too less.

12 -- saving avdsr weights at multipel checkpoints: record_iters = [0, 1e3, 1e4, 2e4, 5e4, 1e5, 2e5, 3e5]
```

To do:
- Regenerate plots for option 1 and option 2 (least priority)
- P OOD, R chaning
	- Save network weights for avdsr option 1, init with option 2.
	- Save network weights for avdsr option 1. 
	- Use these to train various networks and plot the performance

Done:

- Keep goals in the same room, might give better performance for APSF
	- Nope! This won't work as long as we're doing unsupervised exploration. Need an alternative strategy. This also pinpoints to one of the problems with this approach. Learning the SR is hard because of poor steady state distribution (I think the same problem leads to catasforgettingin DQN, does PER fix catastrophic forgetting? need to check)
