import logging
import random
from datetime import datetime

import numpy as np

from environment.maze import Action, Cell, Maze
from models import AbstractModel


class ValueIterationModel(AbstractModel):
    """Planning-based model that solves the maze via value iteration."""

    def __init__(self, game, **kwargs):
        super().__init__(game, name="ValueIterationModel", **kwargs)
        self.V = dict()
        self.Q = dict()
        self.policy = dict()
        self._states = self._enumerate_states()
        self._exit_cell = self._find_exit_cell()

    def train(self, stop_at_convergence=False, **kwargs):
        """Run value iteration until convergence."""
        discount = kwargs.get("discount", 0.90)
        theta = kwargs.get("theta", 1e-4)
        max_iterations = max(kwargs.get("max_iterations", 1000), 1)

        start_time = datetime.now()
        iterations = 0
        delta_history = []

        for state in self._states:
            self.V[state] = 0.0

        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            for state in self._states:
                if state == self._exit_cell:
                    for action in self.environment.actions:
                        self.Q[(state, action)] = 0.0
                    continue

                old_value = self.V[state]
                q_values = self._state_action_values(state, discount)
                best_value = np.max(q_values) if q_values.size > 0 else 0.0
                self.V[state] = best_value

                for idx, action in enumerate(self.environment.actions):
                    self.Q[(state, action)] = q_values[idx]

                delta = max(delta, abs(old_value - best_value))

            delta_history.append(delta)
            iterations = iteration
            logging.info("iteration: {:d}/{:d} | delta: {:.6f}"
                         .format(iteration, max_iterations, delta))

            self.environment.render_q(self)

            if delta < theta:
                break

        self._update_policy(discount)

        elapsed = datetime.now() - start_time
        logging.info("iterations: {:d} | time spent: {}".format(iterations, elapsed))

        return delta_history, [], iterations, elapsed

    def q(self, state):
        """Return q values for all actions for a certain state."""
        state = self._normalize_state(state)
        q_values = [self.Q.get((state, action), 0.0) for action in self.environment.actions]
        return np.array(q_values)

    def predict(self, state):
        """Choose the greedy action based on the derived policy."""
        state = self._normalize_state(state)
        if state not in self.policy:
            q_values = self.q(state)
            best_actions = np.nonzero(q_values == np.max(q_values))[0]
            return self.environment.actions[random.choice(best_actions)]
        return self.policy[state]

    def _enumerate_states(self):
        states = []
        for cell in self.environment.cells:
            if self.environment.maze[cell[::-1]] != Cell.OCCUPIED:
                states.append(cell)
        return states

    def _find_exit_cell(self):
        for cell in self.environment.cells:
            if (self.environment.maze[cell[::-1]] == Cell.EMPTY and
                    cell not in self.environment.empty):
                return cell
        raise RuntimeError("Exit cell could not be determined.")

    def _possible_actions(self, state):
        col, row = state
        possible = self.environment.actions.copy()

        nrows, ncols = self.environment.maze.shape
        if row == 0 or (row > 0 and self.environment.maze[row - 1, col] == Cell.OCCUPIED):
            possible.remove(Action.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.environment.maze[row + 1, col] == Cell.OCCUPIED):
            possible.remove(Action.MOVE_DOWN)
        if col == 0 or (col > 0 and self.environment.maze[row, col - 1] == Cell.OCCUPIED):
            possible.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.environment.maze[row, col + 1] == Cell.OCCUPIED):
            possible.remove(Action.MOVE_RIGHT)

        return possible

    def _transition(self, state, action):
        if action not in self._possible_actions(state):
            return state, Maze.penalty_impossible_move

        col, row = state
        if action == Action.MOVE_LEFT:
            col -= 1
        elif action == Action.MOVE_RIGHT:
            col += 1
        elif action == Action.MOVE_UP:
            row -= 1
        elif action == Action.MOVE_DOWN:
            row += 1

        next_state = (col, row)

        if next_state == self._exit_cell:
            reward = Maze.reward_exit
        else:
            reward = Maze.penalty_move

        return next_state, reward

    def _state_action_values(self, state, discount):
        q_values = []
        for action in self.environment.actions:
            next_state, reward = self._transition(state, action)
            value = reward + discount * self.V.get(next_state, 0.0)
            q_values.append(value)
        return np.array(q_values)

    def _update_policy(self, discount):
        for state in self._states:
            if state == self._exit_cell:
                for action in self.environment.actions:
                    self.Q[(state, action)] = 0.0
                continue

            q_values = self._state_action_values(state, discount)
            for idx, action in enumerate(self.environment.actions):
                self.Q[(state, action)] = q_values[idx]
            best_actions = np.nonzero(q_values == np.max(q_values))[0]
            if best_actions.size > 0:
                chosen = random.choice(best_actions)
                self.policy[state] = self.environment.actions[chosen]
            else:
                self.policy[state] = random.choice(self.environment.actions)

    @staticmethod
    def _normalize_state(state):
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        return state