# GridWorld Reinforcement Learning

Reinforcement Learning implementation of GridWorld in [Reinforcement Learning: An Introduction, p72](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) and [Markov Decision Process and Exact Solution Methods](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec2-mdps-exact-methods.pdf). The following descibes the environment and the agent:

- The agent (robot) lives in a grid
- Walls block the agent’s path
- The agent’s actions do not always go as planned:
    - 80% of the time, the action North takes the agent North (if there is no wall there)
    - 10% of the time, North takes the agent West; 10% East
    - If there is a wall in the direction the agent would have been taken, the agent stays put