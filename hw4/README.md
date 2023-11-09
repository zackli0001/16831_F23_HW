## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions.

2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmuroboticsdrl/16831_F23_HW/blob/main/hw4/cs16831/hw4_part1/scripts/run_hw4_mb.ipynb) **Part I (Model Based Learning)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmuroboticsdrl/16831_F23_HW/blob/main/hw4/run_hw4_expl.ipynb) **Part I (Exploration)**

## Complete the code

### Part 1
The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from Piazza'.

- [cs16831/hw4_part1/infrastructure/rl_trainer.py](cs16831/hw4_part1/infrastructure/rl_trainer.py)
- [cs16831/hw4_part1/infrastructure/utils.py](cs16831/hw4_part1/infrastructure/utils.py)

You will then need to implement code in the following files:
- [cs16831/hw4_part1/agents/mb_agent.py](cs16831/hw4_part1/agents/mb_agent.py)
- [cs16831/hw4_part1/models/ff_model.py](cs16831/hw4_part1/models/ff_model.py)
- [cs16831/hw4_part1/policies/MPC_policy.py](cs16831/hw4_part1/policies/MPC_policy.py)
- [cs16831/hw4_part1/infrastructure/rl_trainer.py](cs16831/hw4_part1/infrastructure/rl_trainer.py)
- [cs16831/hw4_part1/agents/mbpo_agent.py](cs16831/hw4_part1/infrastructure/rl_trainer.py)

The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw4_mb.py](cs16831/hw4_part1/scripts/run_hw4_mb.py) and [scripts/run_hw4_mbpo.py](cs16831/hw4_part1/scripts/run_hw4_mbpo.py) (if running locally) or [scripts/run_hw4_mb.ipynb](cs16831/hw4_part1/scripts/run_hw4_mb.ipynb) and [scripts/run_hw4_mbpo.ipynb](cs16831/hw4_part1/scripts/run_hw4_mbpo.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

### Part 2
The following files have blanks to be filled with your solutions from homework 1 and 3. The relevant sections are marked with `TODO'. You can get solutions from Ed. 

- [cs16831/hw4_part2/infrastructure/utils.py](cs16831/hw4_part2/infrastructure/utils.py)
- [cs16831/hw4_part2/infrastructure/rl_trainer.py](cs16831/hw4_part2//infrastructure/rl_trainer.py)
- [cs16831/hw4_part2/policies/MLP_policy.py](cs16831/hw4_part2/policies/MLP_policy.py)
- [cs16831/hw4_part2/policies/argmax_policy.py](cs16831/hw4_part2//policies/argmax_policy.py)
- [cs16831/hw4_part2/critics/dqn_critic.py](cs16831/hw4_part2/critics/dqn_critic.py)

You will then need to implement code in the following files:

For RND :
- [cs16831/hw4_part2/exploration/rnd_model.py](cs16831/hw4_part2/exploration/rnd_model.py)
- [cs16831/hw4_part2/agents/explore_or_exploit_agent.py](cs16831/hw4_part2/agents/explore_or_exploit_agent.py)

See the [assignment PDF](cs16831_hw4.pdf) for more details on what files to edit.

