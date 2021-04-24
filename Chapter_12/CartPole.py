import gym
env = gym.make('CartPole-v1')
observation = env.reset()
for step_index in range(1000):
    env.render()
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    print("Step {}:".format(step_index))
    print("Action: {}".format(action))
    print("Observation: {}".format(observation))
    print("Reward: {}".format(reward))
    print("Is Done?: {}".format(done))
    print("Info: {}".format(info))
observation = env.reset()
env.close()