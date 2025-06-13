#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import gym_aloha
import time
from collections import deque
import random

class SimpleAgent:
    """シンプルなランダムエージェント"""
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        return self.action_space.sample()

class QLearningAgent:
    """Q-Learningエージェント（簡単な実装）"""
    def __init__(self, action_space, state_size=10, learning_rate=0.1, epsilon=0.1):
        self.action_space = action_space
        self.action_size = action_space.shape[0]
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = {}
        
    def discretize_state(self, observation):
        """連続状態を離散化"""
        if isinstance(observation, dict):
            # agent_posを使用
            state = observation['agent_pos']
        else:
            state = observation
        
        # 状態を離散化（簡単な実装）
        discrete_state = tuple(np.round(state * 10).astype(int))
        return discrete_state
    
    def get_action(self, observation):
        discrete_state = self.discretize_state(observation)
        
        # ε-greedy方策
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, observation, action, reward, next_observation, done):
        discrete_state = self.discretize_state(observation)
        next_discrete_state = self.discretize_state(next_observation)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        if next_discrete_state not in self.q_table:
            self.q_table[next_discrete_state] = np.zeros(self.action_size)
        
        # Q-Learning更新
        current_q = self.q_table[discrete_state][action]
        next_max_q = np.max(self.q_table[next_discrete_state])
        new_q = current_q + self.learning_rate * (reward + 0.99 * next_max_q * (1 - done) - current_q)
        self.q_table[discrete_state][action] = new_q

def train_with_random_agent():
    """ランダムエージェントでの学習"""
    print("ランダムエージェントで学習を開始します...")
    
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    agent = SimpleAgent(env.action_space)
    
    episode_rewards = []
    total_steps = 0
    
    for episode in range(100):  # 100エピソード
        observation, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(1000):  # 最大1000ステップ
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            total_steps += 1
            
            observation = next_observation
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"エピソード {episode}: 平均報酬 = {avg_reward:.3f}, ステップ数 = {step_count}")
    
    env.close()
    print(f"学習完了！最終平均報酬: {np.mean(episode_rewards[-10:]):.3f}")
    return episode_rewards

def train_with_qlearning():
    """Q-Learningエージェントでの学習"""
    print("Q-Learningエージェントで学習を開始します...")
    
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    agent = QLearningAgent(env.action_space)
    
    episode_rewards = []
    total_steps = 0
    
    for episode in range(200):  # 200エピソード
        observation, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(1000):  # 最大1000ステップ
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Q-Learning更新
            agent.update(observation, action, reward, next_observation, terminated)
            
            episode_reward += reward
            step_count += 1
            total_steps += 1
            
            observation = next_observation
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"エピソード {episode}: 平均報酬 = {avg_reward:.3f}, ステップ数 = {step_count}")
    
    env.close()
    print(f"学習完了！最終平均報酬: {np.mean(episode_rewards[-20:]):.3f}")
    return episode_rewards

def record_demo():
    """デモ動画の録画"""
    print("デモ動画を録画します...")
    
    import imageio
    
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    observation, info = env.reset()
    frames = []
    
    for step in range(500):  # 500ステップ
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        image = env.render()
        frames.append(image)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
    
    # 動画を保存
    imageio.mimsave("demo.mp4", np.stack(frames), fps=25)
    print("デモ動画を demo.mp4 に保存しました")

if __name__ == "__main__":
    print("ALOHA環境での学習を開始します")
    print("1. ランダムエージェント")
    print("2. Q-Learningエージェント")
    print("3. デモ動画録画")
    
    choice = input("選択してください (1-3): ")
    
    if choice == "1":
        train_with_random_agent()
    elif choice == "2":
        train_with_qlearning()
    elif choice == "3":
        record_demo()
    else:
        print("無効な選択です。ランダムエージェントで学習を開始します。")
        train_with_random_agent() 