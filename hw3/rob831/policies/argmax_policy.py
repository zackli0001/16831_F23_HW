import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maximizes the Q-value 
        # at the current observation as the output
        # HINT1: you can use self.critic.qa_values
        # HINT2: you can use np.argmax

        qa_values = self.critic.qa_values(observation) # shape: (batch_size, num_actions)
        action = np.argmax(qa_values, axis=1)  # shape: (batch_size,)

        return action.squeeze()
