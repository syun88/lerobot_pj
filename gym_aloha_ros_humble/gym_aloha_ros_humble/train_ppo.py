#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import gym_aloha
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
from gymnasium import spaces

class AgentPosOnlyWrapper(gym.ObservationWrapper):
    """エージェント位置のみを使用するラッパー"""
    
    def __init__(self, env):
        super().__init__(env)
        self._set_obs_space()
    
    def _set_obs_space(self):
        """観測空間をエージェント位置のみに設定"""
        if isinstance(self.observation_space, spaces.Dict):
            if "agent_pos" in self.observation_space.spaces:
                self.observation_space = self.observation_space["agent_pos"]
    
    def observation(self, obs):
        """エージェント位置のみを返す"""
        if isinstance(obs, dict) and "agent_pos" in obs:
            return obs["agent_pos"]
        return obs

class ActionScalingWrapper(gym.ActionWrapper):
    """アクションをスケーリングして物理シミュレーションを安定化"""
    
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale
    
    def action(self, action):
        """アクションをスケーリング"""
        return action * self.scale

def make_env():
    """環境を作成する関数"""
    def _make_env():
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
        env = AgentPosOnlyWrapper(env)
        env = ActionScalingWrapper(env, scale=0.05)  # アクションを小さくスケーリング
        return env
    return _make_env

def train_ppo():
    """PPOで学習を実行"""
    print("PPOで学習を開始します...")
    
    # 環境を作成
    env = DummyVecEnv([make_env() for _ in range(1)])  # 1つの環境
    
    # 評価用環境
    eval_env = DummyVecEnv([make_env() for _ in range(1)])
    
    # 評価コールバック
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # PPOモデルを作成（より安定なパラメータ）
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,  # より小さな学習率
        n_steps=1024,  # より小さなステップ数
        batch_size=32,  # より小さなバッチサイズ
        n_epochs=5,  # より少ないエポック数
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # より小さなクリップ範囲
        tensorboard_log="./tensorboard_logs/",
        device="cpu"  # CPUを使用して安定性を向上
    )
    
    # 学習を実行
    print("学習を開始します...")
    model.learn(
        total_timesteps=100000,  # より少ないステップ数で開始
        callback=eval_callback,
        progress_bar=True
    )
    
    # モデルを保存
    model.save("ppo_aloha_final")
    print("学習完了！モデルを ppo_aloha_final に保存しました")
    
    env.close()
    eval_env.close()
    
    return model

def test_model(model_path="ppo_aloha_final"):
    """学習済みモデルをテスト"""
    print(f"モデル {model_path} をテストします...")
    
    # モデルを読み込み
    model = PPO.load(model_path)
    
    # テスト環境を作成
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="human")
    env = AgentPosOnlyWrapper(env)
    env = ActionScalingWrapper(env, scale=0.05)
    
    total_rewards = []
    
    for episode in range(5):  # 5エピソードテスト
        observation, info = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"エピソード {episode + 1}: 報酬 = {episode_reward:.3f}")
    
    env.close()
    print(f"テスト完了！平均報酬: {np.mean(total_rewards):.3f}")

def record_trained_model(model_path="ppo_aloha_final"):
    """学習済みモデルでデモ動画を録画"""
    print(f"モデル {model_path} でデモ動画を録画します...")
    
    import imageio
    
    # モデルを読み込み
    model = PPO.load(model_path)
    
    # 録画用環境を作成
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    env = AgentPosOnlyWrapper(env)
    env = ActionScalingWrapper(env, scale=0.05)
    observation, info = env.reset()
    frames = []
    
    for step in range(500):  # 500ステップ
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        image = env.render()
        frames.append(image)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
    
    # 動画を保存
    imageio.mimsave("trained_demo.mp4", np.stack(frames), fps=25)
    print("デモ動画を trained_demo.mp4 に保存しました")

if __name__ == "__main__":
    print("PPO学習スクリプト")
    print("1. PPOで学習")
    print("2. 学習済みモデルをテスト")
    print("3. 学習済みモデルでデモ録画")
    
    choice = input("選択してください (1-3): ")
    
    if choice == "1":
        train_ppo()
    elif choice == "2":
        test_model()
    elif choice == "3":
        record_trained_model()
    else:
        print("無効な選択です。PPOで学習を開始します。")
        train_ppo() 