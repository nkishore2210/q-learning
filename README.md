# Q Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
→ Initialize Q-table and hyperparameters.<br>
→ Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.<br>
→ After training, derive the optimal policy from the Q-table.<br>
→ Implement the Monte Carlo method to estimate state values.<br>
→ Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.<br>

## Q LEARNING FUNCTION
#### Name: KISHORE N
#### Register Number: 212222240049
```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
          action = select_action(state, Q, epsilon[e])
          next_state, reward, done, _ = env.step(action)
          td_target = reward + gamma * Q[next_state].max() * (not done)
          td_error = td_target - Q[state][action]
          Q[state][action] = Q[state][action] + alphas[e] * td_error
          state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
### Optimal State Value Functions:

<img width="1236" height="172" alt="image" src="https://github.com/user-attachments/assets/dfefd1f1-3ae2-48bd-887a-e74d85501817" />

### Optimal Action Value Functions:

<img width="1342" height="674" alt="image" src="https://github.com/user-attachments/assets/bd35ecff-a8b9-4f84-ab3a-332c6e4770cd" />

### State value functions of Monte Carlo method:

<img width="1625" height="730" alt="image" src="https://github.com/user-attachments/assets/4e949324-c8f1-4efd-91c0-f90f42fb393f" />

### State value functions of Qlearning method:

<img width="1673" height="730" alt="image" src="https://github.com/user-attachments/assets/d3bcc9fb-eb6f-49d6-8e25-064513866414" />



## RESULT:
Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
