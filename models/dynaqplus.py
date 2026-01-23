import logging
import random
from datetime import datetime
import math

import numpy as np

from environment import Status
from models import AbstractModel
from models import bfs

# The Dyna-Q Algorithm from Intro to RL by Sutton & Barto
# Initialize Q(s, a) and Model(s, a) for all s ∈ S and a ∈ A(s)
# Do forever:
# (a) S ← current (nonterminal) state
# (b) A ← -greedy(S, Q)
# (c) Execute action A; observe resultant reward, R, and state, S′
# (d) Q(S, A) ← Q(S, A) + α[R + γ maxa Q(S′, a) − Q(S, A)]
# (e) M odel(S, A) ← R, S′ (assuming deterministic environment)
# (f) Repeat n times:
# S ← random previously observed state
# A ← random action previously taken in S
# R, S′ ← M odel(S, A)
# Q(S, A) ← Q(S, A) + α[R + γ maxa Q(S′, a) − Q(S, A)

# Dyna-Q+ described in Intro to RL by Sutton & Barto
# Keep track of states' last visited #, introduce a small bonus R + κ√τ to its reward during simulation



class DynaQPlusModel(AbstractModel):
    
    """ Tabular Q-learning prediction model.

        For every state (here: the agents current location ) the value for each of the actions is stored in a table.
        The key for this table is (state + action). Initially all values are 0. When playing training games
        after every move the value in the table is updated based on the reward gained after making the move. Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).
    """
    default_check_convergence_every = 5  # by default check for convergence every # episodes

    def __init__(self, game, **kwargs):
        """ Create a new prediction model for 'game'.

        :param class Maze game: Maze game object
        :param kwargs: model dependent init parameters
        """
        super().__init__(game, name="DynaQPlusModel", **kwargs)
        self.Q = dict()  # table with value for (state, action) combination
        self.model = {} # I realized I was resetting dqModel with every train... initiate model here instead
        self.lastVisited = {}
        self.t = 0 # store global step counter instead of resetting every train

        
    def train(self, stop_at_convergence=False, **kwargs):
        """ Train the model.

            :param stop_at_convergence: stop training as soon as convergence is reached

            Hyperparameters:
            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)
        epsilon_min = 0.02
        time_weight = kwargs.get("time_weight", 1e-3) # new parameter weight for time bonus
        start_cell = kwargs.get("start_cell", (0, 0))

        # for Dyna-Q model, store knowledge for learning
        dqModel = self.model
        # for storing states' last visited num
        lastVisited = self.lastVisited
        # number of planning iterations
        n_planning = kwargs.get("n_planning", 10)
        # variables for reporting purposes
        cumulative_reward_history = []
        win_history = []
        metrics = []
        cumulative_steps = 0
        cumulative_planning = 0

        start_list = list()
        start_time = datetime.now()

        # training starts here
        for episode in range(1, episodes + 1):

            # add counters for metrics
            steps = 0
            planning_updates = 0
            replay_states = set()
            explore_count = 0
            greedy_count = 0
            

            # optimization: make sure to start from all possible cells
            # if not start_list:
            #     start_list = self.environment.empty.copy()
            # start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple so it can be used as dictionary key
            episode_reward = 0.0
            while True:
                
                # choose action epsilon greedy (off-policy, instead of only using the learned policy)
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                    explore_count += 1 # Add 1 to explore
                else:
                    action = self.predict(state)
                    greedy_count += 1 # Add 1 to exploit (greedy)

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                steps += 1 # Add count to step
                cumulative_steps += 1

                episode_reward += reward

                if (state, action) not in self.Q.keys():  # ensure value exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0

                if status in (Status.WIN, Status.LOSE):
                    target = reward
                else:
                    max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])
                    target = reward + discount * max_next_Q

                self.Q[(state, action)] += learning_rate * (target - self.Q[(state, action)])

                dqStatus = status in (Status.WIN, Status.LOSE)

                # Start of Dyna-Q algorithm, add observed state and its reward, state transition
                # into the Dyna-Q internal model
                dqModel[(state,action)] = (next_state, reward, dqStatus)
                
                # Dyna-Q+ add-on --------
                # Iterate global step count
                self.t += 1
                # When visited, set lastVisited to current global step count, meaning it was last visited on this step
                lastVisited[(state,action)] = self.t

                # Implement Dyna-Q's planning to simulate random experience
                for _ in range(n_planning):

                    # get random state from internal model
                    # retrieve its stored reward, state transition
                    # sim_done = if that state reached terminal state (win, lose)
                    s, a = random.choice(list(dqModel.keys()))
                    next_s, sim_reward, sim_done = dqModel[(s, a)]

                    # Dyna-Q+ add on, add the time bonus using the formula time_weight * sqrt last_visited value
                    if (s, a) not in lastVisited:
                        age = 0
                    else:
                        age = self.t - lastVisited[(s, a)]
                    sim_reward += time_weight * math.sqrt(age)

                    # if terminal state, no further actions taken. no need to estimate next value
                    if sim_done:
                        sim_target = sim_reward
                    # get the best estimated value of the next state 
                    else:
                        sim_next_Q = max([self.Q.get((next_s, a2), 0.0) for a2 in self.environment.actions])
                        sim_target = sim_reward + discount * sim_next_Q
                    # update internal model using q-learning update rule
                    prev = self.Q.get((s, a), 0.0)
                    self.Q[(s, a)] = prev + learning_rate * (sim_target - prev)

                    planning_updates += 1 # Add count to planning
                    cumulative_planning += 1
                    replay_states.add(s) # Track states replayed

                

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop training episode
                    break



                state = next_state

            
            cumulative_reward_history.append(episode_reward)
            success = int(status == Status.WIN)

            # Store metrics
            metrics.append({
                "episode": episode,
                "return_": episode_reward,
                "steps": steps,
                "success": success,
                "epsilon": exploration_rate,
                "explore_count": explore_count,
                "n_planning": n_planning,
                "greedy_count": greedy_count,
                "planning_updates": planning_updates,
                "cumulative_steps": cumulative_steps,
                "cumulative_planning_steps": cumulative_planning,
                "cumulative_total_updates": cumulative_steps + cumulative_planning,
                "model_size": len({s for (s, _) in dqModel.keys()}), 
                "replay_unique_states": len(replay_states)
            })

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            # if episode % check_convergence_every == 0:
            #     # check if the current model does win from all starting cells
            #     # only possible if there is a finite number of starting states
            #     w_all, win_rate = self.environment.check_win_all(self)
            #     total_score = 0
            #     valid_cases = 0
            #     # iterate through k number of samples
            #     for start_cell in random.sample(self.environment.empty, k=min(10, len(self.environment.empty))):
            #         # call bfs to find the optimal path
            #         bfs_len = bfs.bfs_compute(self.environment, start_cell)
            #         # goal not found, skip
            #         if np.isinf(bfs_len):
            #             continue 
            #         steps_est = compute_path_length(self, self.environment, start_cell)
            #         if np.isfinite(steps_est):
            #             # find the score of model's prediction
            #             score = bfs_len / steps_est
            #             total_score += score
            #             valid_cases += 1

                        

            #     if valid_cases > 0:
            #         avg_score = total_score / valid_cases
            #     else:
            #         avg_score = 0.0
                
            #     logging.info(f"episode {episode}: win_rate={win_rate:.2f}, optimality={avg_score:.2f}")

            #     win_history.append((episode, win_rate))

            #     if avg_score > 0.9 and win_rate > 0.95 and stop_at_convergence:
            #         break

            #     if w_all is True and stop_at_convergence is True:
            #         logging.info("won from all start cells, stop learning")
            #         break

            exploration_rate = max(epsilon_min, exploration_rate * exploration_decay)  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time, metrics
    
    


    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: game state
            :return int: selected action
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)
    
def compute_path_length(model, environment, start):
        # this is to compute path length as metric for comparison with BFS for optimal convergence policy
        state = environment.reset(start)
        steps = 0
        while True:
            # predict based on model's learnt optimal path
            action = model.predict(state)
            next_state, _, status = environment.step(action)
            steps += 1

            if status in (Status.WIN, Status.LOSE):
                break

            state = next_state

            if steps > environment.maze.size * 2:
                return float('inf')
            
        return steps if status == Status.WIN else float('inf')