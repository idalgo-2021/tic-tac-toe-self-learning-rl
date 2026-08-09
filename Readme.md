# Tic-Tac-Toe. Self-Learning with Reinforcement Learning

Several reinforcement learning (**RL**) methods are implemented for training a neural network (multilayer perceptron) to play Tic-Tac-Toe through self-play.

*[Read this document in Russian](Readme_RU.md)*


## Features

* Neural network architectures:

  * TD methods and MC: `9 → hidden(27) → 1`, `18 → hidden(27) → 1`
  * Q-learning: `18 → hidden(36) → 1`
* Implementation: Python + NumPy
* Training: self-play with a manual implementation of backpropagation
* RL methods:

  * Backward TD(0)
  * Batch TD(0)
  * Online TD(0)
  * Monte Carlo
  * Q-learning
* Saving and loading trained models
* Playing against the trained neural network

## Training

```bash
# Start self-play training (default number of episodes)
# python3 [filename] train
python3 tic_tac_toe_backwardTD.py train  # example


# To specify the number of training episodes, use the --episodes option
# python3 [filename] train --episodes 30000
```

## Playing Against the Network

```bash
# Start an interactive game against the trained network
# python3 [filename] play
python3 tic_tac_toe_backwardTD.py play  # example


# To specify the model to use, provide the model filename with the --load option
# python3 [filename] play --load model.npz
```
