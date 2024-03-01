"""Pseudocode for PPO RL training."""

import torch
import torch.optim as optim
import numpy as np
import Environment

# for exo
TRAJECTORY_FIELDS_EXO = [
	'state',   # State of the agent.
	'action',  # Action taken by the agent.
	'reward',  # Reward for the agent after taking the step.
	'value',   # value of the value network
	'logits',  # logits of the action network.
]

# for human
TRAJECTORY_FIELDS_HUMAN = [
	'state',   # State of the agent.
	'action',  # Action taken by the agent.
	'reward',  # Reward for the agent after taking the step.
	'value',   # value of the value network
	'logits',  # logits of the action network.
]

# for muscle
TRAJECTORY_FIELDS_MUSCLES = [
	'JtA', 
	'tau_des',    # desired tau
	'L', 
	'b',
]

MAX_ITERATION = 100000
REPLAY_BUFFER_SIZE = 30000
NUM_AGENT = 16

class PPO(object):
	def __init__(self,):
		# create multiple simulation environments
		self.env = EnvManager(Environment, NUM_AGENT)
		# build models for exoskeleton, human, and human muscle
		self.exo_model = SimulationNN(self.num_state,self.num_action)
		self.human_model = SimulationHumanNN(self.num_human_state, self.num_human_action)
		self.muscle_model = MuscleNN(self.MuscleRelatedDofs, self.num_muscles)

		# create replay buffer for  exoskeleton, human, and human muscle
		self.replay_exo_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, TRAJECTORY_FIELDS_EXO)
		self.replay_human_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, TRAJECTORY_FIELDS_HUMAN)
		self.replay_muscle_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, TRAJECTORY_FIELDS_MUSCLES)

		self.batch_size = 128
		self.learning_rate = 1e-4

		# create optimizer for exoskeleton, human, and human muscle models
		self.optimizer_exo = optim.Adam(self.exo_model.parameters(),lr=self.learning_rate) 
		self.optimizer_human = optim.Adam(self.human_model.parameters(),lr=self.learning_rate)
		self.optimizer_muscle =  optim.Adam(self.muscle_model.parameters(),lr=self.learning_rate)

		self.episodes = [None]*NUM_AGENT
		for j in range(NUM_AGENT):
			self.episodes[j] = EpisodeBuffer()

		self.gamma = 0.99
		self.lb = 0.99
		self.w_entropy = -0.001


	def ComputeTDandGAE(self):
		self.replay_exo_buffer.Clear()
		self.replay_human_buffer.Clear()
		self.replay_muscle_buffer.Clear()

		for epi in self.total_episodes:
			data = epi.GetData()
			
			states_exo, actions_exo, rewards_exo, values_exo, logprobs_exo, \
			states_human, actions_human, rewards_human, values_human, logprobs_human = zip(*data)

			# Here omit how to compute Temporal Difference(TD) and Generalized Advantage Estimation (GAE),
			# we follows the common way as discribed in PPO: https://arxiv.org/pdf/1707.06347.pdf

			self.replay_exo_buffer.Push(states_exo[i], actions_exo[i], logprobs_exo[i], TD_exo[i], advantages_exo[i])
			self.replay_human_buffer.Push(states_human[i], actions_human[i], logprobs_human[i], TD_human[i], advantages_human[i])
			
			muscle_tuples = self.env.GetMuscleTuples()
			for i in range(len(muscle_tuples)):
				self.muscle_buffer.Push(muscle_tuples[i][0],muscle_tuples[i][1],muscle_tuples[i][2],muscle_tuples[i][3])


	def GenerateTransitions(self):
		self.total_episodes = []
		rewards_exo = [None]*NUM_AGENT
		rewards_human = [None] * NUM_AGENT

		local_step = 0
		while True:
			# Get observation/state of exoskeleton and human from simulators.
			states_exo = self.env.GetExoObservations()
			states_human = self.env.GetHumanObservations()

			# Predict and Apply action of human
			action_dist_human, values_human = self.human_model(states_human)
			actions_human = action_dist_human.sample()
			logprobs_human = action_dist_human.log_prob(actions_human)
			self.env.SetHumanActions(actions_human)

			# Predict and Apply action of exo
			action_dist_exo, values_exo = self.exo_model(states_exo)
			actions_exo = action_dist_exo.sample()
			logprobs_exo = action_dist_exo.log_prob(actions_exo)
			self.env.SetExoActions(actions_exo)

			# Predict and Apply action of muscle
			
			muscle_torque = self.env.GetMuscleTorques()
			desired_torque = self.env.GetDesiredTorquesHuman()
			activations = self.muscle_model(muscle_torque, desired_torque)
			self.env.SetActivationLevels(activations)
			self.env.Steps()

			for j in range(NUM_AGENT):
				# check if the episode of jth agent ends
				if not self.env.IsEndOfEpisode(j):
					# Obtain the rewards after taking action in the simulator
					rewards_exo[j] = self.env.GetExoReward(j) 
					rewards_human[j] = self.env.GetHumanReward(j)
					
					self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j], \
											states_human[j], actions_human[j], rewards_human[j], values_human[j], logprobs_human[j])

					local_step += 1
	
				else:
					self.total_episodes.append(self.episodes[j])
					self.episodes[j] = EpisodeBuffer()
					self.env.Reset(j)


	def GetLoss(self, a_dist, value, action, lp, td, gae):
		'''Critic Loss'''
		loss_critic = ((value-td).pow(2)).mean()
		'''Actor Loss'''
		ratio = torch.exp(a_dist.log_prob(action)-lp)
		gae = (gae-gae.mean())/(gae.std()+ 1E-5)
		loss_actor = (ratio * gae).mean()
		'''Entropy Loss'''
		loss_entropy = - self.w_entropy * a_dist.entropy().mean()
		loss = loss_actor + loss_critic + loss_entropy
		return loss

	def OptimizeSimulationExoNN(self):
		all_transitions = np.array(self.replay_exo_buffer.buffer)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				state, action, lp, td, gae = transitions
				a_dist, value = self.exo_model(state)
				loss = self.GetLoss(a_dist, value, action, lp, td, gae)
				self.optimizer_exo.zero_grad()
				loss.backward()
				self.optimizer_exo.step()

	def OptimizeSimulationHumanNN(self):
		all_transitions = np.array(self.replay_human_buffer.buffer)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				state, action, lp, td, gae = transitions
				a_dist, value = self.human_model(state)
				loss = self.GetLoss(a_dist, value, action, lp, td, gae)
				self.optimizer_human.zero_grad()
				loss.backward()
				self.optimizer_human.step()

	def OptimizeMuscleNN(self):
		muscle_transitions = np.array(self.muscle_buffer.buffer)
		for j in range(self.num_epochs_muscle):
			np.random.shuffle(muscle_transitions)
			for i in range(len(muscle_transitions)//self.muscle_batch_size):
				tuples = muscle_transitions[i*self.muscle_batch_size:(i+1)*self.muscle_batch_size]
				JtA, tau_des, L, b  = tuples
				activation = self.muscle_model(JtA, tau_des)
				tau = torch.einsum('ijk,ik->ij',(L, activation)) + b
				loss_reg = (activation).pow(2).mean()
				loss_target = (((tau-stack_tau_des)/100.0).pow(2)).mean() 
				loss = 0.01*loss_reg + loss_target
				self.optimizer_muscle.zero_grad()
				loss.backward()
				self.optimizer_muscle.step()

	def Train(self):		
		# generate transition
		self.GenerateTransitions()
		# prepare data
		self.ComputeTDandGAE()
		# optimize each model
		self.OptimizeSimulationNN()
		self.OptimizeSimulationHumanNN()
		self.OptimizeMuscleNN()


if __name__=="__main__":
	ppo = PPO()
	for i in range(MAX_ITERATION):
		ppo.Train()


		