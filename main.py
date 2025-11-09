import logging
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import random


import models
from environment.maze import Maze, Render

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level


class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    RANDOM_MODEL = auto()
    VALUE_ITERATION = auto()
    Q_LEARNING = auto()
    Q_ELIGIBILITY = auto()
    SARSA = auto()
    SARSA_ELIGIBILITY = auto()
    DEEP_Q = auto()
    LOAD_DEEP_Q = auto()
    SPEED_TEST_1 = auto()
    SPEED_TEST_2 = auto()
    DYNA_Q = auto()
    Q_LEARNING_COMPARISON = auto()
    DYNA_Q_VS_QL_STATIC = auto() # For Dyna-Q comparison with Q-Learning with static maze
    DYNA_Q_VS_QL_DYNAMIC = auto() # For Dyna-Q comparison with Q-Learning with dynamic maze


test = Test.SHOW_MAZE_ONLY # which test to run

maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]
])  # 0 = free, 1 = occupied

maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])  # 0 = free, 1 = occupied

comparison_maze = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 0]
])

comparison_maze_dynamic = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 0]
])


game = Maze(comparison_maze)
comparison_game1 = Maze(comparison_maze)
dynamic_game1 = Maze(comparison_maze)
dynamic_game2 = Maze(comparison_maze_dynamic)
# dq_game1 = Maze(comparison_maze)
# dq_game2 = Maze(dq_maze2)

# only show the maze
if test == Test.SHOW_MAZE_ONLY:
    game.render(Render.MOVES)
    game.reset()

# play using random model
if test == Test.RANDOM_MODEL:
    game.render(Render.MOVES)
    model = models.RandomModel(game)
    game.play(model, start_cell=(0, 0))

# plan using value iteration
if test == Test.VALUE_ITERATION:
    game.render(Render.TRAINING)
    model = models.ValueIterationModel(game)
    h, w, _, _ = model.train(discount=0.90, theta=1e-4, max_iterations=1000)

if test == Test.DYNA_Q_VS_QL_STATIC:
    

    q_model = models.QTable2CModel(comparison_game1)
    dq_model = models.DynaQModel(comparison_game1)

    comparison_game1.render(Render.NOTHING)
    h1, w1, _, _, q_metrics = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )

    print("Showing learned path for DQ_Maze1 (Q-learning)...")
    maze_q = Maze(comparison_game1.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    comparison_game1.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )
    print("Showing learned path for DQ_Maze1 (Dyna-Q)...")
    maze_dq = Maze(comparison_game1.maze.copy())
    maze_dq.render(Render.MOVES)
    maze_dq.play(dq_model, start_cell=(0, 0))

    # Plot metrics
    q_episodes = [m["episode"] for m in q_metrics]
    q_returns = [m["return_"] for m in q_metrics]
    q_steps = [m["steps"] for m in q_metrics]
    q_success = [m["success"] for m in q_metrics]
    q_cumulative_steps = [m["cumulative_steps"] for m in q_metrics]

    dq_episodes = [m["episode"] for m in dq_metrics]
    dq_returns = [m["return_"] for m in dq_metrics]
    dq_steps = [m["steps"] for m in dq_metrics]
    dq_success = [m["success"] for m in dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in dq_metrics]
    dq_cumulative_total_updates = [m["cumulative_total_updates"] for m in dq_metrics]

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, label="Total Rewards for Q-Learning")
    plt.plot(dq_episodes, dq_returns, label="Total Rewards for Dyna-Q")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning vs Dyna-Q comparison: Total Rewards")
    plt.legend()

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, label="Steps Taken for Q-Learning")
    plt.plot(dq_episodes, dq_steps, label="Steps Taken for Dyna-Q")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title("Q-Learning vs Dyna-Q comparison: Steps Taken")
    plt.legend()

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].plot(dq_episodes, dq_cumulative_steps, label="Exploration Steps")
    axes[0].set_title("Dyna-Q: Exploration Steps")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Steps")
    axes[0].legend()
    axes[1].plot(dq_episodes, dq_cumulative_total_updates, label="Total Steps (Exploration + Simulated)")
    axes[1].set_title("Dyna-Q: Total Steps: Exploration + Simulated")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative Total Updates")
    axes[1].legend()

    plt.show()



if test == Test.DYNA_Q_VS_QL_DYNAMIC:
    q_model = models.QTable2CModel(Maze(dynamic_game1.maze.copy()))
    dq_model = models.DynaQModel(Maze(dynamic_game1.maze.copy()))
    
    episodes_before_change = 150

    dynamic_game1.render(Render.NOTHING)
    h1, w1, _, _, q_metrics_before = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False
    )
    print("Showing learned path before Change (Q-learning)...")
    maze_q = Maze(dynamic_game1.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    dq_model.environment.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics_before = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    print("Showing learned path before Change (Dyna-Q)...")
    maze_dq = Maze(dynamic_game1.maze.copy())
    maze_dq.render(Render.MOVES)
    maze_dq.play(dq_model, start_cell=(0, 0))

    # Test VI
    vi_model_static = models.ValueIterationModel(Maze(dynamic_game1.maze.copy()))
    vi_model_static.train(discount=0.9)
    print("Showing optimal path (Value Iteration) - Original Maze:")
    maze_vi1 = Maze(dynamic_game1.maze.copy())
    maze_vi1.render(Render.MOVES)
    maze_vi1.play(vi_model_static, start_cell=(0, 0))
    

    # Change Maze Environment ------------------------------------------
    q_model.environment = Maze(comparison_maze_dynamic.copy())
    dq_model.environment = Maze(comparison_maze_dynamic.copy())

    h3, w3, _, _, q_metrics_after = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False
    )
    print("Showing learned path after Change (Q-learning)...")
    q_model.environment.render(Render.MOVES)
    q_model.environment.play(q_model, start_cell=(0, 0))

    h4, w4, _, _, dq_metrics_after = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    print("Showing learned path after Change (Dyna-Q)...")
    dq_model.environment.render(Render.MOVES)
    dq_model.environment.play(dq_model, start_cell=(0, 0))

    # VI CHANGED
    vi_model_dynamic = models.ValueIterationModel(Maze(comparison_maze_dynamic.copy()))
    vi_model_dynamic.train(discount=0.9)
    print("Showing optimal path (Value Iteration) - Changed Maze:")
    maze_vi_fail = Maze(comparison_maze_dynamic.copy())
    maze_vi_fail.render(Render.MOVES)
    maze_vi_fail.play(vi_model_static, start_cell=(0, 0))

    for m in q_metrics_after:
        m["episode"] += episodes_before_change
    for m in dq_metrics_after:
        m["episode"] += episodes_before_change
    q_metrics = q_metrics_before + q_metrics_after
    dq_metrics = dq_metrics_before + dq_metrics_after



    # Plot metrics
    q_episodes = [m["episode"] for m in q_metrics]
    q_returns = [m["return_"] for m in q_metrics]
    q_steps = [m["steps"] for m in q_metrics]
    q_success = [m["success"] for m in q_metrics]
    q_cumulative_steps = [m["cumulative_steps"] for m in q_metrics]

    dq_episodes = [m["episode"] for m in dq_metrics]
    dq_returns = [m["return_"] for m in dq_metrics]
    dq_steps = [m["steps"] for m in dq_metrics]
    dq_success = [m["success"] for m in dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in dq_metrics]
    dq_cumulative_total_updates = [m["cumulative_total_updates"] for m in dq_metrics]

    def roll(y, w=25):
        y = np.asarray(y, float)
        return np.convolve(y, np.ones(w)/w, mode='valid')
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, label="Total Rewards for Q-Learning")
    plt.plot(dq_episodes, dq_returns, label="Total Rewards for Dyna-Q")
    plt.plot(q_episodes[len(q_episodes)-len(roll(q_returns)):], roll(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(roll(dq_returns)):], roll(dq_returns), label="Dyna-Q (smoothed)")
    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.text(episodes_before_change+10, np.min(q_returns), "Maze Change", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning vs Dyna-Q comparison: Total Rewards")
    plt.legend()

    

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.35, label="Steps Q-Learning (raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.35, label="Steps Dyna-Q (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(roll(q_steps)):], roll(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(roll(dq_steps)):], roll(dq_steps), label="Dyna-Q (smoothed)")
    plt.axvline(episodes_before_change, color='red', ls='--', label="Maze Changed")
    plt.xlabel("Episode"); plt.ylabel("Steps Taken"); plt.title("Q-Learning vs Dyna-Q: Steps")
    plt.legend()

    plt.show()

    # def recovery_episodes(returns, t_change, w=25, frac=0.95):
    #     base = np.mean(returns[max(0,t_change-w):t_change])
    #     after = roll(returns[t_change:], w)
    #     meet = np.where(after >= frac*base)[0]
    #     return None if len(meet)==0 else int(meet[0])

    # rec_q  = recovery_episodes(q_returns,  episodes_before_change)
    # rec_dq = recovery_episodes(dq_returns, episodes_before_change)
    # print("Episodes to 95% recovery  |  Q-Learning:", rec_q, "  Dyna-Q:", rec_dq)


# train using tabular Q-learning
if test == Test.Q_LEARNING:
    game.render(Render.TRAINING)
    model = models.QTableModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)
    
# train using Dyna-Q
# if test == Test.DYNA_Q:
    

# train using tabular Q-learning and an eligibility trace (aka TD-lambda)
if test == Test.Q_ELIGIBILITY:
    game.render(Render.NOTHING)
    model = models.QTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular SARSA learning
if test == Test.SARSA:
    game.render(Render.NOTHING)
    model = models.SarsaTableModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular SARSA learning and an eligibility trace
if test == Test.SARSA_ELIGIBILITY:
    game.render(Render.NOTHING)  # shows all moves and the q table; nice but slow.
    model = models.SarsaTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# # train using a neural network with experience replay (also saves the resulting model)
# if test == Test.DEEP_Q:
#     game.render(Render.NOTHING)
#     model = models.QReplayNetworkModel(game)
#     h, w, _, _ = model.train(discount=0.80, exploration_rate=0.10, episodes=maze.size * 10, max_memory=maze.size * 4,
#                              stop_at_convergence=True)

# draw graphs showing development of win rate and cumulative rewards
try:
    h  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.manager.set_window_title(model.name)
    if w:
        ax1.plot(*zip(*w))
        ax1.set_xlabel("episode")
        ax1.set_ylabel("win rate")
    else:
        ax1.set_axis_off()
        ax1.text(0.5, 0.5, "win rate unavailable", ha="center", va="center", transform=ax1.transAxes)
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass

# load a previously trained model
if test == Test.LOAD_DEEP_Q:
    model = models.QReplayNetworkModel(game, load=True)

# compare learning speed (cumulative rewards and win rate) of several models in a diagram
if test == Test.SPEED_TEST_1:
    rhist = list()
    whist = list()
    names = list()

    models_to_run = [0, 1, 2, 3, 4]

    for model_id in models_to_run:
        logging.disable(logging.WARNING)
        if model_id == 0:
            model = models.QTableModel(game)
        elif model_id == 1:
            model = models.SarsaTableModel(game)
        elif model_id == 2:
            model = models.QTableTraceModel(game)
        elif model_id == 3:
            model = models.SarsaTableTraceModel(game)
        elif model_id == 4:
            model = models.QReplayNetworkModel(game)

        r, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, exploration_decay=0.999, learning_rate=0.10,
                                 episodes=300)
        rhist.append(r)
        whist.append(w)
        names.append(model.name)

    f, (rhist_ax, whist_ax) = plt.subplots(2, len(models_to_run), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(rhist)):
        rhist_ax[i].set_title(names[i])
        rhist_ax[i].set_ylabel("cumulative reward")
        rhist_ax[i].plot(rhist[i])

    for i in range(len(whist)):
        whist_ax[i].set_xlabel("episode")
        whist_ax[i].set_ylabel("win rate")
        whist_ax[i].plot(*zip(*(whist[i])))

    plt.show()

# run a number of training episodes and plot the training time and episodes needed in histograms (time-consuming)
if test == Test.SPEED_TEST_2:
    runs = 10

    epi = list()
    nme = list()
    sec = list()

    models_to_run = [0, 1, 2, 3, 4]

    for model_id in models_to_run:
        episodes = list()
        seconds = list()

        logging.disable(logging.WARNING)
        for r in range(runs):
            if model_id == 0:
                model = models.QTableModel(game)
            elif model_id == 1:
                model = models.SarsaTableModel(game)
            elif model_id == 2:
                model = models.QTableTraceModel(game)
            elif model_id == 3:
                model = models.SarsaTableTraceModel(game)
            elif model_id == 4:
                model = models.QReplayNetworkModel(game)

            _, _, e, s = model.train(stop_at_convergence=True, discount=0.90, exploration_rate=0.10,
                                     exploration_decay=0.999, learning_rate=0.10, episodes=1000)

            print(e, s)

            episodes.append(e)
            seconds.append(s.seconds)

        logging.disable(logging.NOTSET)
        logging.info("model: {} | trained {} times | average no of episodes: {}| average training time {}"
                     .format(model.name, runs, np.average(episodes), np.sum(seconds) / len(seconds)))

        epi.append(episodes)
        sec.append(seconds)
        nme.append(model.name)

    f, (epi_ax, sec_ax) = plt.subplots(2, len(models_to_run), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(epi)):
        epi_ax[i].set_title(nme[i])
        epi_ax[i].set_xlabel("training episodes")
        epi_ax[i].hist(epi[i], edgecolor="black")

    for i in range(len(sec)):
        sec_ax[i].set_xlabel("seconds per episode")
        sec_ax[i].hist(sec[i], edgecolor="black")

    plt.show()

game.render(Render.MOVES)
# game.play(model, start_cell=(4, 1))

plt.show()  # must be placed here else the image disappears immediately at the end of the program
