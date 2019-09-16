import gym
import numpy as np
import random
from OpenAI_REINFORCE_FC_TF import OpenAI_REINFORCE_FC

env = gym.make('LunarLander-v2')
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
AI_player = OpenAI_REINFORCE_FC(num_actions = num_actions, num_observations = num_states)

gamma_safe = 0.95 # safety threshold
gamma_reward = 1 # haven't done anything with this yet

#! My current implementation only depends on the state
#! if safe, let's pi_perf play, if unsafe gives to pi_safe with
#! prob gamma_safe. However, here the agent will reliably visit 
#! unsafe states, so we ultimately would like to improve it
def h(state, action, next_state=None):
	x = state[0]
	y = state[1]
	theta = state[4]
	tilt = abs(theta) - 0.1 # if craft is tilting too much
	bound = max(-1.8*x - y - 0.2, 1.8*x - y - 0.2) # if device is going out of bounds
	return max(tilt, bound)

#! A point here. We said that the h function is known, thus H_safe and H_perf
#! are known. However, we the dynamics of the environment are unknown (nevermind
#! that they are stochastic in this environment), thus we could not compute it
#! during runtime to select our alpha. So how would we make h depend on the next state?
def H_safe(state):
	return h(state, pi_safe(state))

def H_perf(state):
	return h(state, pi_perf(state))

# Currently returns constant probability determining
# likelyhood of each policy getting selected
def alpha(state):
	#return 0.0 # can see individual behavior of policies by setting to 1 or 0
	H = h(state, None)
	# print(H) # You can uncomment to see the returned safety values
	if H <= 0:
		return 0
	else:
		return gamma_safe

	# This code is currently not executed
	if H_perf <= 0: # Performing policy is safe
		return 0
	else:
		if H_safe > 0:
			return gamma_safe
		else:
			return gamma_safe #! What exactly do we do if they are both unsafe?


# The safe function stabilizes the lateral movement of the
# craft and keeps it at a constant height (improved)
def pi_safe(state):
	if state[3] < -0.3 or (state[3] < 0.04 and state[1] <= 1.1): 
		return 2
	if state[4] < -0.2:
		return 1
	if state[4] > 0.2:
		return 3
	if state[4] < state[2] and state[5] < 0.1:
		return 1
	if state[4] > state[2] and state[5] > - 0.1:
		return 3
	else:
		return 0

# Was able to find a pre-trained agent online and adapt it to our problem
def pi_perf(state):
	return AI_player.AI_Action(observation = state)

# PS. It's also good that we chose probabilities instead of
# outputs for interpolating, since in this example we have a
# discrete action space, hence no interpolation is possible
def pi_blend(state, action_safe, action_perform): #! I suppose the action_safe, action_perform arguments are
												  #! redundant if only going to depend on state
	p = alpha(state)
	
	if random.random() <= p:
		return action_safe
	else:
		return action_perform

# Keep playing until interrupted
episode = 0
while True:
	env.reset()
	action = 0 # Arbitrarily chose no action as first action
	time = 0
	total_reward = 0
	episode += 1
	while True:
		env.render() # Comment to stop rendering
		state, reward, done, info = env.step(action)
		total_reward += reward
		time += 1
	
		action_safe = pi_safe(state) # output of safe policy
		action_perform = pi_perf(state) # output of performing policy
		action = pi_blend(state, action_safe, action_perform) # final action resulting from the blending

		if done:
			print("Episode: {}, Time: {}, Total Reward: {}".format(episode, time, total_reward))
			break

