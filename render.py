# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
import gym
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "test", force=True)
env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    print('hi')
    if done:
        break

env.close()