# Ms. Pac-Man DQN experiment reflection

## Configuration

- Exploration: `0.20`
- Episodes: `100`
- Learning rate: `0.0001`
- Device: Apple MPS
- Python: `3.13.15`
- Environment: `ALE/MsPacman-v5`
- Seed: `42`

I expected modest improvement because 20% exploration preserves opportunities to discover new moves after the random warm-up, while 100 episodes provide repeated experience and a learning rate of 0.0001 keeps the network updates relatively small.

## Results

The run completed all 100 episodes, 63,943 agent decisions, and 15,736 learning updates in about 217 seconds of training and periodic demonstrations.

| Evaluation seed | Before training | After training | Change |
|---:|---:|---:|---:|
| 101 | 350 | 860 | +510 |
| 202 | 500 | 450 | -50 |
| 303 | 320 | 780 | +460 |
| 404 | 800 | 780 | -20 |
| 505 | 490 | 800 | +310 |
| **Mean** | **492** | **734** | **+242** |

The matched-seed evaluation mean increased by about 49%. Three of five games improved, while two declined slightly. The first 25 training episodes averaged 706.4 points and the final 25 averaged 790.8, although individual training scores remained volatile, ranging from 120 to 2,110.

In the seed-101 gameplay excerpts, the untrained agent finished with 350 points. The trained agent moved farther through the maze, collected more pellets, and finished with 860 points.

![Training dashboard](training_dashboard.png)

### Before training

![Untrained gameplay excerpt](episode_0000.gif)

### After training

![Trained gameplay excerpt](final_best.gif)

## Interpretation and limitation

This run provides evidence that learning improved performance under this small, fixed evaluation: the same seeds and 5% evaluation exploration were used before and after, the mean increased, and the saved model weights changed while remaining finite. The rising prediction loss does not by itself prove failure or success; game score is the relevant outcome, and the scores stayed noisy.

The largest limitation is the five-game evaluation sample. It is too small to establish reliable general performance, especially because Atari uses sticky actions and GPU execution can vary even with fixed seeds. This classroom model also uses a deliberately small replay memory and far less training than benchmark-scale DQN agents.

## Next experiment

I would change only the episode count from 100 to 200, keeping exploration at 0.20 and the learning rate at 0.0001. I would then compare the same five evaluation seeds again to test whether longer training produces a more consistent improvement rather than relying on a single favorable run.
