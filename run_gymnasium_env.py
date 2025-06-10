import gymnasium
import gymnasium_env
from gymnasium_env.wrappers import RelativePosition
from gymnasium.wrappers import FlattenObservation


env = gymnasium.make('gymnasium_env/GridWorld-v0', size=5)

observation, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # losowa akcja
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward, observation, info)
    done = terminated or truncated

env.close()