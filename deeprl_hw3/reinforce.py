import gym
import numpy as np


def get_total_reward(env, model):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    # observation is equivalent to state in the last project
    state = env.reset()[0]
    total_reward = 0.0
    done = False

    # Loop until the episode is finished
    while not done:
        # Get the action from the choose_action function
        _, action = choose_action(model, state)
        # Take the action in the environment
        next_state, reward, done, info, _ = env.step(action)
        # Add the reward to the total_reward
        total_reward += reward
    
    return total_reward


def choose_action(model, observation):
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    """
    probability = model.predict(observation)
    # choose an action according to the probability array
    action = np.random.choice(len(probabilities), p = probability)
    # obtain probability of the action
    p = probabilities[action]
    return p, action
    


def reinforce(env,model):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """

    while converge is False:
        episode_data = []
        state = env.reset()[0]
        done = False
        # play a full episode
        while not done:
            p, action = choose_action(model, state)
            next_state, reward, done, *_ = env.step(action)
            episode_data.append((state, action, reward, next_state))
            state = next_state
        
        for t in range(len(episode_data)):
            Gt = sum(gamma**i * episode_data[t+i][2] for i in range(len(episode_data)-t))
            state, action, _, _ = episode_data[t]
            
        model.update(state, action, alpha, Gt)

    total_reward = get_total_reward(env, model)

    return total_reward