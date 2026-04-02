# Reinforcement Learning

### Definition
Reinforcement Learning (RL) is an interactive approach for learning. The learning process can be modeled as a Markovian Decision Process (MDP). This means we can define an state, an action, an agent, an environment and a reward. Each interaction is can be referred to as a step. In each step, the agent chooses an action based on propability ditribution called policy and the environment returns a reward and a new state. The agent objective is to maximize the reward. 

### Basic Concepts
For understanding better the RL, it is crucial to know Bellman Equations, which define the expected rewards through all the possible trajectories and are the basic equations to solve RL tasks. The two equations that compose Bellman Equations are value function and state-value function. An optimal policy can be found by finding optimal state-value function.

There is a difference between on and off-policy methods. If behaviour and target policies are the same, it is an on-policy method. Otherwise, it is an off-policy method. Behaviour policy is the policy that interacts with the environment to generate training data. Target policy is the policy the agent wants to learn.

If the environment model is fully known we can use dynamic programming methods, such as policy iteration or value iteration. However, fully modeling the environment is usually impossible. In this case, there are other options such as the Monte Carlo and Temporal Difference. All these methods need to calculate optimal state-action values to obtain optimal policy, reason why they are called value-based. There are other classes of methods called policy gradient and actor critic methods.

### Main methods

Deep Q-learning is the most classic value-based algorithm. It uses temporal difference, deep neural network for estimating the state-action function, and experience-replay - this last one makes him an off policy method.
One key point of DQN for enhancing exploration is an e-greedy approach.

Rainbow is an algorithm with optimal performance that combines double DQN, priotitized experience replay, dueling network, noisy network, multistep learning and distributional DQN.









### Interesting works

##### Deep Reinforcement Learning: A survey
Presents a background in RL and in deep learning. Also gives an overview about the three DRL algorithms: value-based, policy-based and max-entropy-based.  
[Paper Link](https://ieeexplore.ieee.org/document/9904958)