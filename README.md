Note: originally forked from https://github.com/erikdelange/Reinforcement-Learning-Maze


# RL-Maze-Solver for CSCI323 Modern AI Group Assignment

Goal of this setup is to evaluate the different algorithms learning efficiency, and adaptability to dynamic maze setups. It is extended from https://github.com/erikdelange/Reinforcement-Learning-Maze.

## Algorithms actually implemented/edited by us:
- **Value Iteration (VI)** value_iteration.py - model-based algorithm  (Done by Calvin)
- **Breadth-first search (BFS)** bfs.py - baseline to compute optimal steps for comparison
- **Dyna-Q** dynaq.py - hybrid approach combining model-based & model-free, adapted from Reinforcement Learning: An Intro by Sutton & Barto (2015). (Done by Jia Rong)
- **Dyna-Q+** dynaqplus.py - hybrid approach combining model-based & model-free with added time-based exploration bonus, adapted from Reinforcement Learning: An Intro by Sutton & Barto (2015). (Done by Jia Rong)
- **Q-learning edited** qtable2comparison.py - edited Q-learning code so that metrics can be compared fairly with Dyna-Q (Done by Jia Rong)

## Scenarios Tested
- Static Maze (Q-Learning vs Dyna-Q) - baseline learning efficiency test for algorithms (Done by Jia Rong)
- Dynamic Blocking Maze adapted from Sutton & Barto (2015) - to evaluate if algorithms can adapt when new obstacles block the optimal path (Done by Jia Rong)
- Dynamic Shortcut Maze adapted from Sutton & Barto (2015) - to evaluate if algorithms can adapt when new shortcut appears(Done by Jia Rong)
- Multi-Phase Dynamic Maze (combines essence of blocking and shortcut maze) - evaluate adaptability of algorithms to a continuous changing environment (Done by Jia Rong)

## Environment Setup

### Prerequisites 
- Python version used: 3.10
- IDE used: VSCode

## Install dependencies
pip install numpy matplotlib pandas

# Run the experiments:

All experiments are defined in main.py. To run the experiments, simply find the variable test = Test.MULTI_PHASE_TESTING in main.py and change the variable after 'Test.' to below named variables.

1. Static Maze Experiment
test = Test.DYNA_Q_VS_QL_STATIC

2. Dynamic Blocking Maze (Q-Learning vs Dyna-Q)
test = Test.DYNA_Q_VS_QL_DYNAMIC

3. Dynamic Blocking Maze (Q-Learning vs Dyna-Q vs Dyna-Q+)
test = Test.DYNA_Q_PLUS_BLOCKING

4. Dynamic Shortcut Maze (Q-Learning vs Dyna-Q)
test = Test.DYNA_Q_SHORTCUT

5. Dynamic Shortcut Maze (Q-Learning vs Dyna-Q vs Dyna-Q+)
test = Test.DYNA_Q_PLUS_SHORTCUT

6. Multi-Phase Dynamic Maze (Q-Learning vs Dyna-Q vs Dyna-Q+)
test = Test.MULTI_PHASE_TESTING

Blocking and Shortcut maze (Maze B1, B2, S1, S2, S3 from the report) are defined in mazesetup.py. B1 = comparison_maze, B2 = comparison_maze_dynamic, S1-S3 = shortcut_maze1 - shortcut_maze3. (Designed by Jia Rong)

## References
Sutton, R. S., & Barto, A. G. (2015). Reinforcement Learning: An Introduction (2nd ed.).
https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf







### Escape from a maze using reinforcement learning

##### Solving an optimization problem using an MDP and TD learning

The environment for this problem is a maze with walls and a single exit. An agent (the learner and decision maker) is placed somewhere in the maze. The agents' goal is to reach the exit as quickly as possible. To get there the agent moves through the maze in a succession of steps. For every step the agent must decide which action to take. The options are move left, right, up or down. For this purpose the agent is trained; it learns a policy (Q) which tells what is the best next move to make. With every step the agent incurs a penalty or - when finally reaching the exit - a reward. These penalties and rewards are the input when training the policy. 

![Maze](https://github.com/erikdelange/Reinforcement-Learning-Maze/blob/master/maze.png)

The values for the penalties and rewards are defined in class *Maze* in *maze.py*:
```python
    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze
```
The policies (or models) used here are based on Sarsa and Q-learning. During training the learning algorithm updates the action-value function Q for each state which is visited. The most preferable action is indicated by the highest value. Updating these values is based on the reward or penalty incurred after the action was taken. With TD-learning a model learns at every step it takes, not only when the exit is reached. However, learning does speed up once the exit has been reached for the first time. 

This project demonstrates different models which learn to move through a maze. Class Maze in file *maze.py* in package *environment* defines the environment including the rules of the game (rewards, penalties). In file *main.py* an example of a maze is defined (but you can create your own) as a np.array. By selecting a value for *test* from enum Test a certain model is trained and can then be used to play a number of games from different starting positions in the maze. When training or playing the agents moves can be plotted by calling Maze.render(Render.MOVES). To also display the progress of training call Maze.render(Render.TRAINING). This visualizes the most preferred action per cell. The higher the value the greener the arrow is displayed.

![Maze](https://github.com/erikdelange/Reinforcement-Learning-Maze/blob/master/bestmove.png)

Package *models* contains the following models:
1. *QTableModel* uses a table to record the value of each (state, action) pair. For a state the highest value indicates the most desirable action. These values are constantly refined during training. This is a fast way to learn a policy.
2. *SarsaTableModel* uses a similar setup as the previous model, but takes less risk during learning (= on-policy learning).
3. *QTableTraceModel* is an extension of the QTableModel. It speeds up learning by keeping track of previously visited state-action pairs, and updates their values as well although with a decaying rate.
4. *SarsaTableTraceModel* is a variant of SarsaTableModel but adds an eligibility trace, just as QTableTraceModel. 
5. *ValueIteration* repeatedly calculates V using the Bellman equation until convergence on the solution or it reaches a pre-determined number of iterations.

Requires matplotlib, numpy, keras and tensorflow.
