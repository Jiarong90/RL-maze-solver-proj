import logging
import random
from datetime import datetime

import numpy as np

from environment import Status
from models import AbstractModel
from models import bfs

# This model is for Q-learning as well. However I have adjusted the convergence evaluation
# check to include comparing against BFS steps ratio. 

class QTable2CModel(AbstractModel):
    
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
        super().__init__(game, name="QTable2CModel", **kwargs)
        self.Q = dict()  # table with value for (state, action) combination

        
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
        start_cell = kwargs.get("start_cell", (0, 0))
        # variables for reporting purposes

        cumulative_reward_history = []
        win_history = []
        metrics = []
        cumulative_steps = 0

        start_list = list()
        start_time = datetime.now()

        # training starts here
        for episode in range(1, episodes + 1):

            # add counters for metrics
            steps = 0
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
                    explore_count += 1
                else:
                    action = self.predict(state)
                    greedy_count += 1

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                steps += 1
                cumulative_steps += 1

                episode_reward += reward

                if (state, action) not in self.Q:  # ensure value exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0

                if status in (Status.WIN, Status.LOSE):
                    target = reward
                else:
                    max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])
                    target = reward + discount * max_next_Q

                self.Q[(state, action)] += learning_rate * (target - self.Q[(state, action)])

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop training episode
                    break



                state = next_state


            cumulative_reward_history.append(episode_reward)
            success = int(status == Status.WIN)

            

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            avg_score = 0.0
            bfs_len = 0.0
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

            #     if avg_score > 0.9 and win_rate > 0.95 and stop_at_convergence:
            #             break
                
            #     logging.info(f"episode {episode}: win_rate={win_rate:.2f}, optimality={avg_score:.2f}")
            #     win_history.append((episode, win_rate))
            #     if w_all is True and stop_at_convergence is True:
            #         logging.info("won from all start cells, stop learning")
            #         break

            exploration_rate = max(epsilon_min, exploration_rate * exploration_decay)  # explore less as training progresses

            # Insert metrics
            metrics.append({
                "episode": episode,
                "return_": episode_reward,
                "steps": steps,
                "success": int(status == Status.WIN),
                "epsilon": exploration_rate,
                "explore_count": explore_count,
                "greedy_count": greedy_count,
                "cumulative_steps": cumulative_steps,
            })

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