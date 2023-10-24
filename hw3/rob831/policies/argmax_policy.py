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

        qa_values = self.critic.qa_values(observation) # shape: (frame_history_len, ac_dim)
        actions_of_all_obs = np.argmax(qa_values.squeeze(), axis=1)  # shape: (frame_history_len,)
        actions, counts = np.unique(actions_of_all_obs, return_counts=True)
        index_of_most_frequent = np.argmax(counts)
        action = actions[index_of_most_frequent]

        return action 
