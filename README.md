# Ms. Pac-Man DQN assignment

## Experiment overview

This project trains a Deep Q-Network (DQN) agent to play `ALE/MsPacman-v5` from game pixels. The final notebook was executed locally from top to bottom on an Apple Silicon Mac using PyTorch's MPS backend. It saved the untrained baseline, 100 training episodes, periodic demonstrations, a trained checkpoint, a five-game final evaluation, plots, logs, GIFs, and a ZIP archive.

The final matched-seed evaluation mean increased from **492** before training to **734** after training, a change of **+242 points** (about **+49.2%**). Three of the five evaluation games improved and two declined slightly, so this is evidence of improvement in this run rather than a guarantee of general performance.

## Final run summary

| Item | Actual final-run value |
|---|---:|
| Exploration | `0.20` |
| Episodes requested and completed | `100` |
| Learning rate | `0.0001` |
| Agent decisions | `63,943` |
| Learning updates | `15,736` |
| Recorded training and periodic-demo time | `216.783 seconds` (`3m 36.8s`) |
| Learning device | Apple MPS |
| Hardware | Apple M5 MacBook Air, 10 cores, 32 GB memory |
| Operating system | macOS 26.6.2, arm64 |
| Python | 3.13.15 |
| Environment | `ALE/MsPacman-v5` |

The recorded elapsed time comes directly from `training_summary.json`. It starts with the training loop and includes periodic demonstrations; the notebook also ran the separate five-game baseline and five-game final evaluations.

## Hyperparameters and prediction

I chose:

- `EXPLORATION = 0.20`
- `EPISODES = 100`
- `LEARNING_RATE = 0.0001`

I expected modest improvement. A 20% exploration rate continues trying random moves after replay warm-up, 100 episodes provide repeated experience, and a learning rate of 0.0001 keeps Adam's weight updates relatively small. These were the assignment's sensible starting values and fit a local MPS run without changing the fixed classroom methodology.

## What the agent sees, does, and receives

The agent observes a stack of **four grayscale 84 × 84 screens**. One screen shows positions; four successive screens also provide information about motion.

At each decision, the network chooses one of the nine available Atari joystick actions: no-op, up, right, left, down, or one of the four diagonal directions. Atari preprocessing advances four emulator frames per agent decision.

The environment returns a game reward after each action. The replay memory stores a clipped training reward between -1 and +1, while the training logs and evaluation results report the original, unclipped Ms. Pac-Man score.

## How DQN learns

The convolutional DQN estimates a future-reward value, or Q-value, for every available action. After the initial 1,000 fully random decisions, the agent chooses a random action 20% of the time to explore and otherwise chooses the action with the largest predicted value.

Replay memory holds up to 5,000 past transitions. Random batches of 32 transitions mix experiences from different moments and reuse them for learning instead of updating only from consecutive gameplay. The model learns every four decisions after warm-up using Adam, Huber loss, and a discount factor of 0.99.

A separate target network supplies a more stable estimate of the next screen's value. The training network's weights are copied to it every 1,000 decisions.

## Evaluation methodology

The baseline is the untrained neural network, not a random-action agent. Before and after training, evaluation used the same five seeds (`101`, `202`, `303`, `404`, and `505`), 5% evaluation exploration, and a maximum of 3,000 decisions per game. Evaluation used a separate environment and did not update weights or replay memory. None of these fixed settings were changed.

| Game / seed | Before training | After training | Change |
|---:|---:|---:|---:|
| 1 / 101 | 350 | 860 | +510 |
| 2 / 202 | 500 | 450 | -50 |
| 3 / 303 | 320 | 780 | +460 |
| 4 / 404 | 800 | 780 | -20 |
| 5 / 505 | 490 | 800 | +310 |
| **Mean** | **492** | **734** | **+242** |

No baseline or trained evaluation game reached the time limit.

## Training evidence

The first 25 training episodes averaged 706.4 points and the final 25 averaged 790.8. Individual scores were volatile, ranging from 120 to 2,110. The loss curve rises during much of the run, but loss alone does not establish gameplay improvement; the fixed before/after scores above are the relevant comparison.

![Training dashboard](results/training_dashboard.png)

### Untrained gameplay

This excerpt is from evaluation seed 101. Its complete untrained game scored 350.

![Untrained Ms. Pac-Man agent](results/episode_0000.gif)

### Periodic gameplay evaluations

Each GIF is a 20-second game-time excerpt played at 4× speed. The caption gives the full-game score from the separate seed-101 demonstration.

#### After episode 25 — full-game score 410

![Ms. Pac-Man agent after 25 episodes](results/episode_0025.gif)

#### After episode 50 — full-game score 320

![Ms. Pac-Man agent after 50 episodes](results/episode_0050.gif)

#### After episode 75 — full-game score 510

![Ms. Pac-Man agent after 75 episodes](results/episode_0075.gif)

#### After episode 100 — full-game score 860

![Ms. Pac-Man agent after 100 episodes](results/episode_0100.gif)

### Final trained gameplay

The best final evaluation was seed 101 with a full-game score of 860. In the excerpt, the trained agent travels farther through the maze and collects more pellets than the untrained version.

![Final trained Ms. Pac-Man agent](results/final_best.gif)

## What I observed

The trained mean was 242 points higher than the untrained mean under the same evaluation conditions. Seeds 101, 303, and 505 improved substantially. Seeds 202 and 404 fell by 50 and 20 points, respectively. The periodic seed-101 results were also non-monotonic—410, 320, 510, and 860—showing that performance did not improve smoothly throughout training.

## Limitation

The strongest limitation is the **five-game evaluation sample**. Two of the five paired scores declined, training scores ranged from 120 to 2,110, and the periodic seed-101 score temporarily dropped at episode 50. These observations show substantial variance, so the +242 mean change should not be treated as a reliable benchmark estimate. Sticky actions and GPU execution can also vary even when seeds are fixed.

## Next experiment

I would change exactly one hyperparameter: increase `EPISODES` from `100` to `200`. I would keep `EXPLORATION = 0.20`, `LEARNING_RATE = 0.0001`, and every fixed evaluation and classroom setting unchanged. This would test whether additional training makes the matched-seed improvement more consistent.

## Run the notebook locally

Use Python 3.11–3.13. From this repository:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt ipykernel
python -m ipykernel install --sys-prefix --name py313 --display-name "Python 3.13 (pacman-dqn)"
python -m jupyter lab pacman_dqn.ipynb
```

Select the `Python 3.13 (pacman-dqn)` / `py313` kernel, confirm the three hyperparameters near the top, and run all cells in order. The notebook automatically tests CUDA, then Apple MPS, then CPU using a real DQN minibatch before starting the experiment.

## Submission files

- [Executed final notebook](pacman_dqn.ipynb)
- [Final configuration](results/config.json)
- [Episode-by-episode training data](results/training.csv)
- [Training summary](results/training_summary.json)
- [Before/after comparison](results/comparison.json)
- [Periodic evaluation scores](results/demo_scores.json)
- [Written reflection](results/reflection.md)

The complete final run, playback checkpoints, and ZIP remain locally under `pacman_runs/`. That directory and all `.pt` files are ignored and are not part of the Git submission.
