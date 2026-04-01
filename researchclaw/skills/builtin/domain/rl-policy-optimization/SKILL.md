---
name: rl-policy-optimization
description: Best practices for reinforcement learning policy optimization. Use when working on RL agents, PPO, SAC, or reward design.
metadata:
  category: domain
  trigger-keywords: "reinforcement learning,rl,policy,reward,agent,environment,ppo,sac"
  applicable-stages: "9,10"
  priority: "3"
  version: "1.0"
  author: researchclaw
  references: "Schulman et al., Proximal Policy Optimization, 2017; Haarnoja et al., Soft Actor-Critic, ICML 2018"
---

## RL Policy Optimization Best Practice
Algorithm selection:
- Discrete actions: PPO, DQN, A2C
- Continuous actions: SAC, TD3, PPO
- Multi-agent: MAPPO, QMIX
- Offline: CQL, IQL, Decision Transformer

Training recipe:
- PPO: clip=0.2, lr=3e-4, gamma=0.99, GAE lambda=0.95
- SAC: lr=3e-4, tau=0.005, auto-tune alpha
- Use vectorized environments (e.g., gymnasium.vector)
- Normalize observations and rewards
- Log episode return, episode length, value loss, policy entropy

Evaluation:
- Report mean +/- std over 10+ evaluation episodes
- Use deterministic policy for evaluation
- Compare against random policy and simple baselines
- Report sample efficiency (return vs. env steps)

Common pitfalls:
- Reward shaping can introduce bias
- Seed sensitivity is HIGH — use 5+ seeds
- Hyperparameter sensitivity — do a small sweep
