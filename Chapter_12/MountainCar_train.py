import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as np_utils
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()
goal_steps = 200
score_requirement = -198
intial_games = 20000

def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            
            if observation[0] > -0.2:
                reward = 1

            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                output = np_utils.to_categorical(data[1], 3)
                training_data.append([data[0], output])
        
        env.reset()

    print(accepted_scores)
    
    return training_data

training_data = model_data_preparation()

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

def train_model(training_data):
    data_x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    data_y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(data_x[0]), output_size=len(data_y[0]))
    
    model.fit(data_x, data_y, epochs=20)
    return model

trained_model = train_model(training_data)

scores = []
choices = []
success_count = 0
for each_game in range(100):
    score = 0
    prev_obs = []
    print('Game {} playing'.format(each_game))
    for step_index in range(goal_steps):
        # Uncomment below line if you want to see how our bot is playing the game.
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0, 3)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score += reward
        if done:
            print('Final step count: {}'.format(step_index + 1))
            if (step_index + 1) < 200:
                # if goal achieved in less than 200 steps, consider successful
                success_count += 1
            break

    env.reset()
    scores.append(score)

print(scores)

# since we ran 100 games, success count is equal to percentage
print('Success percentage: {}%'.format(success_count))  
print('Average Score:', sum(scores)/len(scores))
print('choice 0:{}  choice 1:{}  choice 2:{}'.format(choices.count(0)/len(choices), choices.count(1)/len(choices), choices.count(2)/len(choices)))

# draw the histogram of scores
plt.hist(scores, bins=5)
plt.show()