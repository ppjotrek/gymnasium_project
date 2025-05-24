import gymnasium
import gymnasium_env
from gymnasium_env.wrappers import RelativePosition
from gymnasium.wrappers import FlattenObservation


env = gymnasium.make('gymnasium_env/GridWorld-v0', size=5)
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())

wrapped_env = RelativePosition(env)
print(wrapped_env.reset())  