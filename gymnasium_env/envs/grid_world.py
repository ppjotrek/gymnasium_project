from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.num_walls = 5
        self.num_pits = self.np_random.integers(3, 6) if hasattr(self, 'np_random') else 3  # 3 to 5 pits

        # Add special tiles A and B
        self.special_names = ["A", "B"]

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "walls": spaces.Box(0, size - 1, shape=(self.num_walls, 2), dtype=int),
                "pits": spaces.Box(0, size - 1, shape=(5, 2), dtype=int),  # Max 5 pits
                "A": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "B": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # Uzupełnij pits do 5 elementów
        pits_obs = np.full((5, 2), -1, dtype=int)
        for i, pit in enumerate(self._pits):
            pits_obs[i] = pit.astype(int)
        # Uzupełnij walls do self.num_walls elementów
        walls_obs = np.full((self.num_walls, 2), -1, dtype=int)
        for i, wall in enumerate(self._walls):
            walls_obs[i] = wall.astype(int)
        return {
            "agent": self._agent_location.astype(int),
            "target": self._target_location.astype(int),
            "walls": walls_obs,
            "pits": pits_obs,
            "A": self._A_location.astype(int),
            "B": self._B_location.astype(int),
        }
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place target not on agent
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # Place walls not on agent or target, and not overlapping each other
        self._walls = []
        while len(self._walls) < self.num_walls:
            wall = self.np_random.integers(0, self.size, size=2, dtype=int)
            if (
                not np.array_equal(wall, self._agent_location)
                and not np.array_equal(wall, self._target_location)
                and not any(np.array_equal(wall, w) for w in self._walls)
            ):
                self._walls.append(wall)

        # Place pits not on agent, target, or walls, and not overlapping each other
        self.num_pits = self.np_random.integers(3, 6)
        self._pits = []
        while len(self._pits) < self.num_pits:
            pit = self.np_random.integers(0, self.size, size=2, dtype=int)
            if (
                not np.array_equal(pit, self._agent_location)
                and not np.array_equal(pit, self._target_location)
                and not any(np.array_equal(pit, w) for w in self._walls)
                and not any(np.array_equal(pit, p) for p in self._pits)
            ):
                self._pits.append(pit)
                
        # Place special tiles A and B, not on agent, target, walls, pits, or each other
        self._A_location = self._agent_location
        while (
            np.array_equal(self._A_location, self._agent_location)
            or np.array_equal(self._A_location, self._target_location)
            or any(np.array_equal(self._A_location, w) for w in self._walls)
            or any(np.array_equal(self._A_location, p) for p in self._pits)
        ):
            self._A_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._B_location = self._A_location
        while (
            np.array_equal(self._B_location, self._agent_location)
            or np.array_equal(self._B_location, self._target_location)
            or any(np.array_equal(self._B_location, w) for w in self._walls)
            or any(np.array_equal(self._B_location, p) for p in self._pits)
            or np.array_equal(self._B_location, self._A_location)
        ):
            self._B_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Track if A or B have been visited this episode
        self._visited_A = False
        self._visited_B = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        proposed_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Prevent moving into a wall
        if any(np.array_equal(proposed_location, w) for w in self._walls):
            new_location = self._agent_location
        else:
            new_location = proposed_location

        self._agent_location = new_location

        # Check for pit
        in_pit = any(np.array_equal(self._agent_location, p) for p in self._pits)
        reached_target = np.array_equal(self._agent_location, self._target_location)
        reached_A = np.array_equal(self._agent_location, self._A_location)
        reached_B = np.array_equal(self._agent_location, self._B_location)

        terminated = reached_target or in_pit or (reached_B and not self._visited_B)
        reward = -0.1

        if reached_target:
            reward = 1
        elif in_pit:
            reward = -10
        # Give +10 for first time on A
        if reached_A and not self._visited_A:
            reward += 10
            self._visited_A = True
        # Give +10 for first time on B, and terminate
        if reached_B and not self._visited_B:
            reward += 10
            self._visited_B = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw walls
        for wall in self._walls:
            pygame.draw.rect(
                canvas,
                (128, 128, 128),  # Gray color for walls
                pygame.Rect(
                    pix_square_size * wall,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        for pit in self._pits:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),  # Black color for pits
                pygame.Rect(
                    pix_square_size * pit,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw special tile A (green)
        pygame.draw.rect(
            canvas,
            (0, 200, 0),
            pygame.Rect(
                pix_square_size * self._A_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw special tile B (yellow)
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * self._B_location,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
