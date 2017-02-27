import gym
from gym import wrappers
env = gym.make('SpaceInvaders-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
for i_episode in range(1):
    observation = env.reset()
    for t in range(100000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(done)
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
