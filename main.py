import logging
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import models
import mazesetup

from environment.maze import Maze, Render

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level

np.random.seed(42)
random.seed(42)


class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    VALUE_ITERATION = auto()
    Q_LEARNING = auto()
    Q_ELIGIBILITY = auto()
    SARSA = auto()
    SARSA_ELIGIBILITY = auto()
    # -- Newly added test cases
    DYNA_Q = auto()
    Q_LEARNING_COMPARISON = auto()
    DYNA_Q_VS_QL_STATIC = auto() # For Dyna-Q comparison with Q-Learning with static maze
    DYNA_Q_VS_QL_DYNAMIC = auto() # For Dyna-Q comparison with Q-Learning with dynamic maze
    Q_LEARNING_VS_SARSA = auto()
    DYNA_Q_SHORTCUT = auto()
    DYNA_Q_PLUS_BLOCKING = auto()
    DYNA_Q_PLUS_SHORTCUT = auto()
    DYNA_Q_PLUS_COMPARISON = auto()
    MULTI_PHASE_TESTING = auto()

test = Test.SHOW_MAZE_ONLY# which test to run

mazeType = "blank18" # maze types

if mazeType == "normal8":
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
elif mazeType == "blank8":
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
elif mazeType == "normal18":
    maze = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
elif mazeType == "blank18":
    maze = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])  # 0 = free, 1 = occupied



## ----- Maze setup use for Dyna-Q comparison, I moved mazes to new file mazesetup.py 
#  ----- to reduce clutter
comparison_maze = mazesetup.comparison_maze
comparison_maze_dynamic = mazesetup.comparison_maze_dynamic
shortcut_maze1 = mazesetup.shortcut_maze1
shortcut_maze2 = mazesetup.shortcut_maze2
shortcut_maze3 = mazesetup.shortcut_maze3

game = Maze(shortcut_maze3)

## -- New section added
comparison_game1 = Maze(comparison_maze)
dynamic_game1 = Maze(comparison_maze)
dynamic_game2 = Maze(comparison_maze_dynamic)

# only show the maze
if test == Test.SHOW_MAZE_ONLY:
    game.render(Render.MOVES)
    game.reset((6, 0))

# plan using value iteration
if test == Test.VALUE_ITERATION:
    game.render(Render.TRAINING)
    model = models.ValueIterationModel(game)
    h, w, _, _ = model.train(discount=0.90, theta=1e-4, max_iterations=200)

# train using tabular Q-learning
if test == Test.Q_LEARNING:
    game.render(Render.TRAINING)
    model = models.QTableModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular Q-learning and an eligibility trace (aka TD-lambda)
if test == Test.Q_ELIGIBILITY:
    game.render(Render.TRAINING)
    model = models.QTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular SARSA learning
if test == Test.SARSA:
    game.render(Render.TRAINING)
    model = models.SarsaTableModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular SARSA learning and an eligibility trace
if test == Test.SARSA_ELIGIBILITY:
    game.render(Render.TRAINING)  # shows all moves and the q table; nice but slow.
    model = models.SarsaTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)
    
## --- NEW TEST SETUP SECTION HERE


# SARSA vs Q-Learning
if test == Test.Q_LEARNING_VS_SARSA:
    q_model = models.QTable2CModel(comparison_game1)
    sarsa_model = models.SarsaTableModel(comparison_game1)
    vi_model = models.ValueIterationModel(comparison_game1)

    comparison_game1.render(Render.NOTHING)
    h1, w1, _, _, q_metrics = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )

    maze_q = Maze(comparison_game1.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    comparison_game1.render(Render.NOTHING)
    h2, w2, _, _, sarsa_metrics = sarsa_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )
    maze_sarsa = Maze(comparison_game1.maze.copy())
    maze_sarsa.render(Render.MOVES)
    maze_sarsa.play(sarsa_model, start_cell=(0, 0))

    delta_history, _, vi_iterations, _, vi_reward, vi_steps = vi_model.train()
    maze_vi = Maze(comparison_game1.maze.copy())
    maze_vi.render(Render.NOTHING)
    maze_vi.play(vi_model, start_cell=(0, 0))

    # Plot metrics
    q_episodes = [m["episode"] for m in q_metrics]
    q_returns = [m["return_"] for m in q_metrics]
    q_steps = [m["steps"] for m in q_metrics]
    q_success = [m["success"] for m in q_metrics]

    sarsa_episodes = [m["episode"] for m in sarsa_metrics]
    sarsa_returns = [m["return_"] for m in sarsa_metrics]
    sarsa_steps = [m["steps"] for m in sarsa_metrics]
    sarsa_success = [m["success"] for m in sarsa_metrics]



    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, label="Total Rewards for Q-Learning")
    plt.plot(sarsa_episodes, sarsa_returns, label="Total Rewards for SARSA")
    plt.axhline(y=vi_reward, color='orange', linestyle='--', label=f"Value Iteration (= ≈ {vi_reward:.1f})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("VI vs Q-Learning vs SARSA comparison: Total Rewards")
    plt.legend()

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, label="Steps Taken for Q-Learning")
    plt.plot(sarsa_episodes, sarsa_steps, label="Steps Taken for SARSA")
    plt.axhline(y=vi_steps, color='orange', linestyle='--', label=f"Value Iteration (= {vi_steps})")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title("VI vs Q-Learning vs SARSA comparison: Steps Taken")
    plt.legend()

    plt.figure(figsize=(8,5))
    plt.plot(range(len(delta_history)), delta_history, label="Δ per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Max Change in V")
    plt.title(f"Value Iteration (Converged in {vi_iterations} iterations)")
    plt.yscale("log")  # optional: log scale to show exponential decay
    plt.legend()

    plt.show()

# Static Maze setup for Dyna-Q vs Q-Learning
if test == Test.DYNA_Q_VS_QL_STATIC:
    q_model = models.QTable2CModel(comparison_game1)
    dq_model = models.DynaQModel(comparison_game1)

    comparison_game1.render(Render.NOTHING)
    h1, w1, _, _, q_metrics = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )

    print("Showing learned path for DQ_Maze1 (Q-learning)...")
    maze_q = Maze(q_model.environment.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    comparison_game1.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )
    print("Showing learned path for DQ_Maze1 (Dyna-Q)...")
    maze_dq = Maze(dq_model.environment.maze.copy())
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

    def smooth(y, w=15):
        y = np.asarray(y, float)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean()

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, alpha=0.35, label="Total Rewards for Q-Learning (Raw)")
    plt.plot(dq_episodes, dq_returns, alpha=0.35, label="Total Rewards for Dyna-Q (Raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_returns)):], smooth(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_returns)):], smooth(dq_returns), label="Dyna-Q (smoothed)")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Total Rewards")
    plt.title("Q-Learning vs Dyna-Q comparison: Total Rewards")
    plt.legend()

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.35, label="Steps Taken for Q-Learning (Raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.35, label="Steps Taken for Dyna-Q (Raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_steps)):], smooth(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_steps)):], smooth(dq_steps), label="Dyna-Q (smoothed)")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title("Q-Learning vs Dyna-Q comparison: Steps Taken")
    plt.legend()

    plt.show()


# Dynamic Maze setup for Dyna-Q vs Q-learning (Blocking Scenario) -------------
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
    maze_q = Maze(q_model.environment.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    dq_model.environment.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics_before = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    print("Showing learned path before Change (Dyna-Q)...")
    maze_dq = Maze(dq_model.environment.maze.copy())
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
    q_model.environment.maze = comparison_maze_dynamic.copy()
    q_model.environment.reset((0, 0))
    dq_model.environment.maze = comparison_maze_dynamic.copy()
    dq_model.environment.reset((0, 0))

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

    bfs_before = models.bfs.bfs_compute(Maze(comparison_maze.copy()), start_cell=(0,0))
    bfs_after  = models.bfs.bfs_compute(Maze(comparison_maze_dynamic.copy()), start_cell=(0,0))

    def smooth(y, w=10):
        y = np.asarray(y, float)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean()
    
    # Plot Total Rewards
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, alpha=0.25, label="Total Rewards for Q-Learning (Raw)")
    plt.plot(dq_episodes, dq_returns, alpha=0.25, label="Total Rewards for Dyna-Q (Raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_returns)):], smooth(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_returns)):], smooth(dq_returns), label="Dyna-Q (smoothed)")
    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning vs Dyna-Q comparison: Total Rewards (Blocking Maze)")
    plt.legend()

    # Plot Steps Taken
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.25, label="Steps Q-Learning (raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.25, label="Steps Dyna-Q (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_steps)):], smooth(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_steps)):], smooth(dq_steps), label="Dyna-Q (smoothed)")
    plt.plot(
        [0, episodes_before_change], [bfs_before, bfs_before], color='lime', linestyle='-', label=f"BFS Before = {bfs_before}"
    )
    plt.plot(
        [episodes_before_change, max(q_episodes)], [bfs_after, bfs_after], color='lime', linestyle='-', label=f"BFS After= {bfs_after}"
    )
    plt.axvline(episodes_before_change, color='red', ls='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode"); plt.ylabel("Steps Taken"); plt.title("Q-Learning vs Dyna-Q: Steps Taken (Blocking Maze)")
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




# Dynamic Maze setup for Dyna-Q vs Q-learning (Shortcut Scenario) -----------------------------------
if test == Test.DYNA_Q_SHORTCUT:
    q_model = models.QTable2CModel(Maze(shortcut_maze1.copy()))
    dq_model = models.DynaQModel(Maze(shortcut_maze1.copy()))


    episodes_before_change = 200

    q_model.environment.render(Render.NOTHING)
    h1, w1, _, _, q_metrics_before = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, start_cell=(6, 0)
    )
    print("Showing learned path before Change (Q-learning)...")
    maze_q = Maze(q_model.environment.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(6, 0))

    dq_model.environment.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics_before = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, start_cell=(6, 0)
    )
    print("Showing learned path before Change (Dyna-Q)...")
    maze_dq = Maze(dq_model.environment.maze.copy())
    maze_dq.render(Render.MOVES)
    maze_dq.play(dq_model, start_cell=(6, 0))

    # Change Maze Environment ------------------------------------------
    q_model.environment.maze = shortcut_maze2.copy()
    q_model.environment.reset((6, 0))
    dq_model.environment.maze = shortcut_maze2.copy()
    dq_model.environment.reset((6, 0))

    h4, w4, _, _, q_metrics_after = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, start_cell=(6, 0)
    )
    print("Showing learned path after Change (Q-learning)...")
    q_model.environment.render(Render.MOVES)
    q_model.environment.play(q_model, start_cell=(6, 0))

    h5, w5, _, _, dq_metrics_after = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, start_cell=(6, 0)
    )
    print("Showing learned path after Change (Dyna-Q)...")
    dq_model.environment.render(Render.MOVES)
    dq_model.environment.play(dq_model, start_cell=(6, 0))


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
    q_min_step_before = min(m["steps"] for m in q_metrics_before)
    q_min_step_after = min(m["steps"] for m in q_metrics_after)

    dq_episodes = [m["episode"] for m in dq_metrics]
    dq_returns = [m["return_"] for m in dq_metrics]
    dq_steps = [m["steps"] for m in dq_metrics]
    dq_success = [m["success"] for m in dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in dq_metrics]
    dq_min_step_before = min(m["steps"] for m in dq_metrics_before)
    dq_min_step_after = min(m["steps"] for m in dq_metrics_after)


    bfs_before = models.bfs.bfs_compute(Maze(shortcut_maze1.copy()), start_cell=(6,0))
    bfs_after  = models.bfs.bfs_compute(Maze(shortcut_maze2.copy()), start_cell=(6,0))


    print("Optimal Steps before and after: ----")
    print("Optimal BFS before: ", bfs_before)
    print("Q-Learning before: ", q_min_step_before)
    print("Q-Learning after: ", q_min_step_after)
    print("Optimal BFS after: ", bfs_after)
    print("DQ before: ", dq_min_step_before)
    print("DQ after: ", dq_min_step_after)

    def smooth(y, w=10):
        y = np.asarray(y, float)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean()
    
    # Plot Total Rewards
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, alpha=0.25, label="Total Rewards for Q-Learning (raw)")
    plt.plot(dq_episodes, dq_returns, alpha=0.25, label="Total Rewards for Dyna-Q (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_returns)):], smooth(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_returns)):], smooth(dq_returns), label="Dyna-Q (smoothed)")
    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning vs Dyna-Q comparison: Total Rewards (Shortcut Maze)")
    plt.legend()

    # Plot step size
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.25, label="Steps Q-Learning (raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.25, label="Steps Dyna-Q (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_steps)):], smooth(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_steps)):], smooth(dq_steps), label="Dyna-Q (smoothed)")
    plt.plot(
        [0, episodes_before_change], [bfs_before, bfs_before], color='lime', linestyle='-', label=f"BFS Optimal Steps Before Change = {bfs_before}"
    )
    plt.plot(
        [episodes_before_change, max(q_episodes)], [bfs_after, bfs_after], color='lime', linestyle='-', label=f"BFS Optimal Steps After Change = {bfs_after}"
    )
    plt.axvline(episodes_before_change, color='red', ls='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode"); plt.ylabel("Steps Taken"); plt.title("Q-Learning vs Dyna-Q: Steps (Shortcut Maze)")
    plt.legend()

    plt.show()


# Dynamic Maze setup for Dyna-Q+ vs Dyna-Q vs Q-Learning (Shortcut Scenario) -----------------------------------
if test == Test.DYNA_Q_PLUS_SHORTCUT:
    q_model = models.QTable2CModel(Maze(shortcut_maze1.copy()))
    dq_model = models.DynaQModel(Maze(shortcut_maze1.copy()))
    dqp_model = models.DynaQPlusModel(Maze(shortcut_maze1.copy()))

    episodes_before_change = 200

    q_model.environment.render(Render.NOTHING)
    h1, w1, _, _, q_metrics_before = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, start_cell=(6, 0)
    )
    print("Showing learned path before Change (Q-learning)...")
    maze_q = Maze(q_model.environment.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(6, 0))

    dq_model.environment.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics_before = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, start_cell=(6, 0)
    )
    print("Showing learned path before Change (Dyna-Q)...")
    maze_dq = Maze(dq_model.environment.maze.copy())
    maze_dq.render(Render.MOVES)
    maze_dq.play(dq_model, start_cell=(6, 0))

    dqp_model.environment.render(Render.NOTHING)
    h3, w3, _, _, dqp_metrics_before = dqp_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, start_cell=(6, 0)
    )
    maze_dqp = Maze(dqp_model.environment.maze.copy())
    maze_dqp.render(Render.MOVES)
    maze_dqp.play(dqp_model, start_cell=(6, 0))

    # Change Maze Environment ------------------------------------------
    q_model.environment.maze = shortcut_maze2.copy()
    q_model.environment.reset((6, 0))
    dq_model.environment.maze = shortcut_maze2.copy()
    dq_model.environment.reset((6, 0))
    dqp_model.environment.maze = shortcut_maze2.copy()
    dqp_model.environment.reset((6, 0))

    h4, w4, _, _, q_metrics_after = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, start_cell=(6, 0)
    )
    print("Showing learned path after Change (Q-learning)...")
    q_model.environment.render(Render.MOVES)
    q_model.environment.play(q_model, start_cell=(6, 0))

    h5, w5, _, _, dq_metrics_after = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, start_cell=(6, 0)
    )
    print("Showing learned path after Change (Dyna-Q)...")
    dq_model.environment.render(Render.MOVES)
    dq_model.environment.play(dq_model, start_cell=(6, 0))

    h6, w6, _, _, dqp_metrics_after = dqp_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, time_weight=0.01, start_cell=(6, 0)
    )
    dqp_model.environment.render(Render.MOVES)
    dqp_model.environment.play(dqp_model, start_cell=(6, 0))

    for m in q_metrics_after:
        m["episode"] += episodes_before_change
    for m in dq_metrics_after:
        m["episode"] += episodes_before_change
    for m in dqp_metrics_after:
        m["episode"] += episodes_before_change
    q_metrics = q_metrics_before + q_metrics_after
    dq_metrics = dq_metrics_before + dq_metrics_after
    dqp_metrics = dqp_metrics_before + dqp_metrics_after

    # Plot metrics
    q_episodes = [m["episode"] for m in q_metrics]
    q_returns = [m["return_"] for m in q_metrics]
    q_steps = [m["steps"] for m in q_metrics]
    q_success = [m["success"] for m in q_metrics]
    q_cumulative_steps = [m["cumulative_steps"] for m in q_metrics]
    q_min_step_before = min(m["steps"] for m in q_metrics_before)
    q_min_step_after = min(m["steps"] for m in q_metrics_after)

    dq_episodes = [m["episode"] for m in dq_metrics]
    dq_returns = [m["return_"] for m in dq_metrics]
    dq_steps = [m["steps"] for m in dq_metrics]
    dq_success = [m["success"] for m in dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in dq_metrics]
    dq_min_step_before = min(m["steps"] for m in dq_metrics_before)
    dq_min_step_after = min(m["steps"] for m in dq_metrics_after)

    dqp_episodes = [m["episode"] for m in dqp_metrics]
    dqp_returns = [m["return_"] for m in dqp_metrics]
    dqp_steps = [m["steps"] for m in dqp_metrics]
    dqp_success = [m["success"] for m in dqp_metrics]
    dqp_cumulative_steps = [m["cumulative_steps"] for m in dqp_metrics]
    dqp_min_step_before = min(m["steps"] for m in dqp_metrics_before)
    dqp_min_step_after = min(m["steps"] for m in dqp_metrics_after)

    bfs_before = models.bfs.bfs_compute(Maze(shortcut_maze1.copy()), start_cell=(6,0))
    bfs_after  = models.bfs.bfs_compute(Maze(shortcut_maze2.copy()), start_cell=(6,0))


    print("Optimal Steps before and after: ----")
    print("Optimal BFS before: ", bfs_before)
    print("Q-Learning before: ", q_min_step_before)
    print("Q-Learning after: ", q_min_step_after)
    print("Optimal BFS after: ", bfs_after)
    print("DQ before: ", dq_min_step_before)
    print("DQ after: ", dq_min_step_after)
    print("DQP before: ", dqp_min_step_before)
    print("DQP after: ", dqp_min_step_after)

    def smooth(y, w=10):
        y = np.asarray(y, float)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean()
    
    # Plot Total Rewards
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, alpha=0.25, label="Total Rewards for Q-Learning (raw)")
    plt.plot(dq_episodes, dq_returns, alpha=0.25, label="Total Rewards for Dyna-Q (raw)")
    plt.plot(dqp_episodes, dqp_returns, alpha=0.25, label="Total Rewards for Dyna-Q+ (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_returns)):], smooth(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_returns)):], smooth(dq_returns), label="Dyna-Q (smoothed)")
    plt.plot(dqp_episodes[len(dqp_episodes)-len(smooth(dqp_returns)):], smooth(dqp_returns), label="Dyna-Q+ (smoothed)")
    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Dyna-Q+ vs Dyna-Q vs Q-Learning comparison: Total Rewards (Shortcut Maze)")
    plt.legend()

    # Plot step size
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.25, label="Steps Q-Learning (raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.25, label="Steps Dyna-Q (raw)")
    plt.plot(dqp_episodes, dqp_steps, alpha=0.25, label="Steps Dyna-Q+ (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_steps)):], smooth(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_steps)):], smooth(dq_steps), label="Dyna-Q (smoothed)")
    plt.plot(dqp_episodes[len(dqp_episodes)-len(smooth(dqp_steps)):], smooth(dqp_steps), label="Dyna-Q+ (smoothed)")
    plt.plot(
        [0, episodes_before_change], [bfs_before, bfs_before], color='lime', linestyle='-', label=f"BFS Before = {bfs_before}"
    )
    plt.plot(
        [episodes_before_change, max(q_episodes)], [bfs_after, bfs_after], color='lime', linestyle='-', label=f"BFS After = {bfs_after}"
    )
    plt.axvline(episodes_before_change, color='red', ls='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode"); plt.ylabel("Steps Taken"); plt.title("Dyna-Q+ vs Dyna-Q vs Q-Learning: Steps (Shortcut Maze)")
    plt.legend()

    plt.show()

# Dynamic Maze setup for Dyna-Q+ vs Dyna-Q vs Q-Learning (Blocking Scenario) -----------------------------------
if test == Test.DYNA_Q_PLUS_BLOCKING:
    q_model = models.QTable2CModel(Maze(comparison_maze.copy()))
    dq_model = models.DynaQModel(Maze(comparison_maze.copy()))
    dqp_model = models.DynaQPlusModel(Maze(comparison_maze.copy()))

    episodes_before_change = 150

    q_model.environment.render(Render.NOTHING)
    h1, w1, _, _, q_metrics_before = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False
    )
    print("Showing learned path before Change (Q-learning)...")
    maze_q = Maze(q_model.environment.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    dq_model.environment.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics_before = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    print("Showing learned path before Change (Dyna-Q)...")
    maze_dq = Maze(dq_model.environment.maze.copy())
    maze_dq.render(Render.MOVES)
    maze_dq.play(dq_model, start_cell=(0, 0))

    dqp_model.environment.render(Render.NOTHING)
    h3, w3, _, _, dqp_metrics_before = dqp_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    maze_dqp = Maze(dqp_model.environment.maze.copy())
    maze_dqp.render(Render.MOVES)
    maze_dqp.play(dqp_model, start_cell=(0, 0))

    # Change Maze Environment ------------------------------------------
    q_model.environment.maze = comparison_maze_dynamic.copy()
    q_model.environment.reset((0, 0))
    dq_model.environment.maze = comparison_maze_dynamic.copy()
    dq_model.environment.reset((0, 0))
    dqp_model.environment.maze = comparison_maze_dynamic.copy()
    dqp_model.environment.reset((0, 0))

    h4, w4, _, _, q_metrics_after = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False
    )
    print("Showing learned path after Change (Q-learning)...")
    q_model.environment.render(Render.MOVES)
    q_model.environment.play(q_model, start_cell=(0, 0))

    h5, w5, _, _, dq_metrics_after = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    print("Showing learned path after Change (Dyna-Q)...")
    dq_model.environment.render(Render.MOVES)
    dq_model.environment.play(dq_model, start_cell=(0, 0))

    h6, w6, _, _, dqp_metrics_after = dqp_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, time_weight=0.01
    )
    dqp_model.environment.render(Render.MOVES)
    dqp_model.environment.play(dqp_model, start_cell=(0, 0))

    for m in q_metrics_after:
        m["episode"] += episodes_before_change
    for m in dq_metrics_after:
        m["episode"] += episodes_before_change
    for m in dqp_metrics_after:
        m["episode"] += episodes_before_change
    q_metrics = q_metrics_before + q_metrics_after
    dq_metrics = dq_metrics_before + dq_metrics_after
    dqp_metrics = dqp_metrics_before + dqp_metrics_after

    # Plot metrics
    q_episodes = [m["episode"] for m in q_metrics]
    q_returns = [m["return_"] for m in q_metrics]
    q_steps = [m["steps"] for m in q_metrics]
    q_success = [m["success"] for m in q_metrics]
    q_cumulative_steps = [m["cumulative_steps"] for m in q_metrics]
    q_min_step_before = min(m["steps"] for m in q_metrics_before)
    q_min_step_after = min(m["steps"] for m in q_metrics_after)

    dq_episodes = [m["episode"] for m in dq_metrics]
    dq_returns = [m["return_"] for m in dq_metrics]
    dq_steps = [m["steps"] for m in dq_metrics]
    dq_success = [m["success"] for m in dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in dq_metrics]
    dq_min_step_before = min(m["steps"] for m in dq_metrics_before)
    dq_min_step_after = min(m["steps"] for m in dq_metrics_after)

    dqp_episodes = [m["episode"] for m in dqp_metrics]
    dqp_returns = [m["return_"] for m in dqp_metrics]
    dqp_steps = [m["steps"] for m in dqp_metrics]
    dqp_success = [m["success"] for m in dqp_metrics]
    dqp_cumulative_steps = [m["cumulative_steps"] for m in dqp_metrics]
    dqp_min_step_before = min(m["steps"] for m in dqp_metrics_before)
    dqp_min_step_after = min(m["steps"] for m in dqp_metrics_after)

    bfs_before = models.bfs.bfs_compute(Maze(shortcut_maze1.copy()), start_cell=(0,0))
    bfs_after  = models.bfs.bfs_compute(Maze(shortcut_maze2.copy()), start_cell=(0,0))


    print("Optimal Steps before and after: ----")
    print("Optimal BFS before: ", bfs_before)
    print("Q-Learning before: ", q_min_step_before)
    print("Q-Learning after: ", q_min_step_after)
    print("Optimal BFS after: ", bfs_after)
    print("DQ before: ", dq_min_step_before)
    print("DQ after: ", dq_min_step_after)
    print("DQP before: ", dqp_min_step_before)
    print("DQP after: ", dqp_min_step_after)

    def smooth(y, w=10):
        y = np.asarray(y, float)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean()
    
    # Plot Total Rewards
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, alpha=0.25, label="Total Rewards for Q-Learning (raw)")
    plt.plot(dq_episodes, dq_returns, alpha=0.25, label="Total Rewards for Dyna-Q (raw)")
    plt.plot(dqp_episodes, dqp_returns, alpha=0.25, label="Total Rewards for Dyna-Q+ (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_returns)):], smooth(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_returns)):], smooth(dq_returns), label="Dyna-Q (smoothed)")
    plt.plot(dqp_episodes[len(dqp_episodes)-len(smooth(dqp_returns)):], smooth(dqp_returns), label="Dyna-Q+ (smoothed)")
    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Dyna-Q+ vs Dyna-Q vs Q-Learning comparison: Total Rewards (Blocking Maze)")
    plt.legend()

    # Plot step size
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.25, label="Steps Q-Learning (raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.25, label="Steps Dyna-Q (raw)")
    plt.plot(dqp_episodes, dqp_steps, alpha=0.25, label="Steps Dyna-Q+ (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_steps)):], smooth(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_steps)):], smooth(dq_steps), label="Dyna-Q (smoothed)")
    plt.plot(dqp_episodes[len(dqp_episodes)-len(smooth(dqp_steps)):], smooth(dqp_steps), label="Dyna-Q+ (smoothed)")
    plt.plot(
        [0, episodes_before_change], [bfs_before, bfs_before], color='lime', linestyle='-', label=f"BFS Before = {bfs_before}"
    )
    plt.plot(
        [episodes_before_change, max(q_episodes)], [bfs_after, bfs_after], color='lime', linestyle='-', label=f"BFS After = {bfs_after}"
    )
    plt.axvline(episodes_before_change, color='red', ls='--', label="Maze Changed")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode"); plt.ylabel("Steps Taken"); plt.title("Dyna-Q+ vs Dyna-Q vs Q-Learning: Steps (Blocking Maze)")
    plt.legend()

    plt.show()


# FINAL MULTI-PHASE Dynamic Maze setup for Dyna-Q+ vs Dyna-Q vs Q-Learning (Repeating Scenario) -----------------------------------
if test == Test.MULTI_PHASE_TESTING:

    total_loop_count = 6 # Loop 6 times total
    curr_loop_count = 0 # Current loop count
    maze_count = 3 # Maze scenarios count
    # Initialize models
    q_model = models.QTable2CModel(Maze(shortcut_maze1.copy()))
    dq_model = models.DynaQModel(Maze(shortcut_maze1.copy()))
    dqp_model = models.DynaQPlusModel(Maze(shortcut_maze1.copy()))
    curr_maze = shortcut_maze1
    episodes_before_change = 300

    bfs_steps = []
    training_times = []
    final_q_metrics = []
    final_dq_metrics = []
    final_dqp_metrics = []

    while curr_loop_count < total_loop_count:
        # Implement loop logic, switch between 3 maze states
        if curr_loop_count % maze_count == 0: # First maze scenario
            curr_maze = shortcut_maze1
            
        elif curr_loop_count % maze_count == 1: # Second maze scenario
            curr_maze = shortcut_maze3
            
        elif curr_loop_count % maze_count == 2: #Third maze scenario
            curr_maze = shortcut_maze2

        # Change Maze Environment ------
        q_model.environment.maze = curr_maze.copy()
        q_model.environment.reset((6, 0))
        dq_model.environment.maze = curr_maze.copy()
        dq_model.environment.reset((6, 0))
        dqp_model.environment.maze = curr_maze.copy()
        dqp_model.environment.reset((6, 0))
            
        # Training
        h4, w4, _, q_time, q_metrics = q_model.train(
            discount=0.9, exploration_rate=0.1, learning_rate=0.1,
            episodes=episodes_before_change, stop_at_convergence=False, start_cell=(6, 0)
        )
        print("Showing learned path after Change (Q-learning)...")
        maze_q = Maze(q_model.environment.maze.copy())
        maze_q.render(Render.MOVES)
        maze_q.play(q_model, start_cell=(6, 0))

        h5, w5, _, dq_time, dq_metrics = dq_model.train(
            discount=0.9, exploration_rate=0.1, learning_rate=0.1,
            episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, start_cell=(6, 0)
        )
        print("Showing learned path after Change (Dyna-Q)...")
        maze_dq = Maze(dq_model.environment.maze.copy())
        maze_dq.render(Render.MOVES)
        maze_dq.play(dq_model, start_cell=(6, 0))

        h6, w6, _, dqp_time, dqp_metrics = dqp_model.train(
            discount=0.9, exploration_rate=0.1, learning_rate=0.1,
            episodes=episodes_before_change, stop_at_convergence=False, n_planning=30, time_weight=0.01, start_cell=(6, 0)
        )
        maze_dqp = Maze(dqp_model.environment.maze.copy())
        maze_dqp.render(Render.MOVES)
        maze_dqp.play(dqp_model, start_cell=(6, 0))

        for m in q_metrics:
            m["episode"] += curr_loop_count * episodes_before_change
        for m in dq_metrics:
            m["episode"] += curr_loop_count * episodes_before_change
        for m in dqp_metrics:
            m["episode"] += curr_loop_count * episodes_before_change
        final_q_metrics.extend(q_metrics)
        final_dq_metrics.extend(dq_metrics)
        final_dqp_metrics.extend(dqp_metrics)

        q_seconds = q_time.total_seconds()
        dq_seconds = dq_time.total_seconds()
        dqp_seconds = dqp_time.total_seconds()
        training_times.append({
            "curr_loop": curr_loop_count,
            "maze" : curr_loop_count % maze_count,
            "q_learning": q_seconds,
            "dq": dq_seconds,
            "dqp": dqp_seconds
        })

        bfs_steps.append(models.bfs.bfs_compute(Maze(curr_maze.copy()), start_cell=(6,0)))

        curr_loop_count += 1

    # Plot metrics
    q_episodes = [m["episode"] for m in final_q_metrics]
    q_returns = [m["return_"] for m in final_q_metrics]
    q_steps = [m["steps"] for m in final_q_metrics]
    q_cumulative_steps = [m["cumulative_steps"] for m in final_q_metrics]

    dq_episodes = [m["episode"] for m in final_dq_metrics]
    dq_returns = [m["return_"] for m in final_dq_metrics]
    dq_steps = [m["steps"] for m in final_dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in final_dq_metrics]

    dqp_episodes = [m["episode"] for m in final_dqp_metrics]
    dqp_returns = [m["return_"] for m in final_dqp_metrics]
    dqp_steps = [m["steps"] for m in final_dqp_metrics]
    dqp_cumulative_steps = [m["cumulative_steps"] for m in final_dqp_metrics]

    # Compute average training time taken per algorithm
    avg_time = {
        "Q-Learning": np.mean([t["q_learning"] for t in training_times]),
        "dq": np.mean([t["dq"] for t in training_times]),
        "dqp": np.mean([t["dqp"] for t in training_times])
    }

    # Compute average training time taken per algorithm per maze type
    avg_time_per_maze = {}
    for maze in range(maze_count):
        this_maze_type = []
        for t in training_times:
            if t["maze"] == maze:
                this_maze_type.append(t)
        avg_time_per_maze[maze] = {
            "Q-Learning": np.mean([t["q_learning"] for t in this_maze_type]),
            "dq": np.mean([t["dq"] for t in this_maze_type]),
            "dqp": np.mean([t["dqp"] for t in this_maze_type])
        }

    # Print out stored training times
    print("Overall Average Training Time per algorithm")
    for algo, time in avg_time.items():
        print(f"{algo}: {time}")

    # Print out stored training times per maze
    print("Training Average Training Time per algorithm per maze")
    for maze, times in avg_time_per_maze.items():
        print(f"Maze: {maze}")
        for algo, time in times.items():
            print(f"{algo}: {time}")

    def smooth(y, w=10):
        y = np.asarray(y, float)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean()
    
    # Plot Total Rewards
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, alpha=0.25, label="Total Rewards for Q-Learning (raw)")
    plt.plot(dq_episodes, dq_returns, alpha=0.25, label="Total Rewards for Dyna-Q (raw)")
    plt.plot(dqp_episodes, dqp_returns, alpha=0.25, label="Total Rewards for Dyna-Q+ (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_returns)):], smooth(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_returns)):], smooth(dq_returns), label="Dyna-Q (smoothed)")
    plt.plot(dqp_episodes[len(dqp_episodes)-len(smooth(dqp_returns)):], smooth(dqp_returns), label="Dyna-Q+ (smoothed)")
    for i, changed in enumerate(range(episodes_before_change, total_loop_count * episodes_before_change, episodes_before_change)):
        plt.axvline(changed, color='red', linestyle='--', alpha=0.5, label="Maze Changed" if i == 0 else None)
    plt.ylim(auto=True)
    plt.xlim(0, total_loop_count * episodes_before_change)
    plt.xticks(np.arange(0, (total_loop_count + 1) * episodes_before_change, episodes_before_change))
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Total Rewards")
    plt.title("Dyna-Q+ vs Dyna-Q vs Q-Learning: Total Rewards (Multi-Phase Dynamic Maze)")
    plt.legend()

    # Plot step size
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.25, label="Steps Q-Learning (raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.25, label="Steps Dyna-Q (raw)")
    plt.plot(dqp_episodes, dqp_steps, alpha=0.25, label="Steps Dyna-Q+ (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(smooth(q_steps)):], smooth(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(smooth(dq_steps)):], smooth(dq_steps), label="Dyna-Q (smoothed)")
    plt.plot(dqp_episodes[len(dqp_episodes)-len(smooth(dqp_steps)):], smooth(dqp_steps), label="Dyna-Q+ (smoothed)")
    for i, changed in enumerate(range(episodes_before_change, total_loop_count * episodes_before_change, episodes_before_change)):
        if i < total_loop_count - 1:
            plt.axvline(changed, color='red', linestyle='--', alpha=0.5, label="Maze Changed" if i == 0 else None)   
    episode_start = 0
    for i, changed in enumerate(range(episodes_before_change, total_loop_count * episodes_before_change + episodes_before_change, episodes_before_change)):
        plt.plot(
            [episode_start, changed], [bfs_steps[i], bfs_steps[i]], color='lime', label="BFS Shortest Steps" if i == 0 else None
        )     
        episode_start += episodes_before_change
    plt.ylim(auto=True)
    plt.xlim(0, total_loop_count * episodes_before_change)
    plt.xticks(np.arange(0, (total_loop_count + 1) * episodes_before_change, episodes_before_change))
    plt.grid(alpha=0.3)
    plt.xlabel("Episode"); plt.ylabel("Steps Taken"); plt.title("Dyna-Q+ vs Dyna-Q vs Q-Learning: Steps Taken (Multi-Phase Dynamic Maze)")
    plt.legend()

    plt.show()



## --------------------------------

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

game.render(Render.MOVES)
# game.play(model, start_cell=(4, 1))

plt.show()  # must be placed here else the image disappears immediately at the end of the program
