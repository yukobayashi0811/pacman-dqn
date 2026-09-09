# Ms. Pac-Man DQN experiment reflection

## Configuration

- Exploration: `0.20`
- Episodes: `125`
- Learning rate: `0.00005`
- Device: Apple MPS
- Python: `3.13.15`
- Environment: `ALE/MsPacman-v5`
- Seed: `42`

After short one-variable-at-a-time screens, I expected the lower learning rate to make weight updates steadier and 25 additional episodes to provide enough extra experience, while 20% exploration continued trying alternative actions. A fresh verification run was used rather than reusing candidate weights.

## Results

The run completed all 125 episodes, 76,129 agent decisions, and 18,783 learning updates in 222.543527 seconds of training and periodic demonstrations.

| Evaluation seed | Before training | After training | Change |
|---:|---:|---:|---:|
| 101 | 350 | 710 | +360 |
| 202 | 500 | 840 | +340 |
| 303 | 320 | 790 | +470 |
| 404 | 800 | 660 | -140 |
| 505 | 490 | 710 | +220 |
| **Mean** | **492** | **742** | **+250** |

The matched-seed mean increased by about 50.8%, and four of five games improved. The first 25 training episodes averaged 688.8 points and the final 25 averaged 704.0, though individual scores ranged from 130 to 2,170. The five trained evaluation scores were relatively compact: their standard deviation was about 64.3 and their range was 180.

![Training dashboard](training_dashboard.png)

### Before training

![Untrained gameplay excerpt](episode_0000.gif)

### After training

![Trained gameplay excerpt](final_best.gif)

## Interpretation and limitation

The evaluation supplies evidence of improvement under the fixed classroom methodology: before and after used identical seeds, 5% evaluation exploration, and the same 3,000-decision maximum. Game score, rather than loss, supports the conclusion. Mean episode loss increased late in training even though the trained evaluation mean exceeded the baseline.

The main limitation is that only five fixed games were used, and those same seeds informed hyperparameter selection. The verified mean is only eight points above the previous submission's 734, so that small advantage may not generalize even though the fresh verification reproduced the candidate's scores. Seed 404 also declined after training.

## Next experiment

I would change exactly one hyperparameter: increase episodes from 125 to 150. Exploration would remain 0.20, learning rate would remain 0.00005, and all official evaluation and classroom settings would remain unchanged.
