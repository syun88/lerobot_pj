#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.logger import read_csv

def plot_episode_rewards(rewards, title="学習進捗", save_path=None):
    """エピソード報酬をプロット"""
    plt.figure(figsize=(12, 6))
    
    # 移動平均を計算
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', label=f'{window_size}エピソード移動平均', linewidth=2)
    
    plt.plot(rewards, 'b-', alpha=0.6, label='エピソード報酬')
    plt.xlabel('エピソード')
    plt.ylabel('報酬')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを {save_path} に保存しました")
    
    plt.show()

def plot_training_logs(log_dir="./logs/"):
    """Stable-Baselines3のログをプロット"""
    if os.path.exists(log_dir):
        try:
            plot_results([log_dir], 1e5, "timesteps", "PPO")
            plt.show()
        except Exception as e:
            print(f"ログのプロットに失敗しました: {e}")
    else:
        print(f"ログディレクトリ {log_dir} が見つかりません")

def analyze_training_results(log_dir="./logs/"):
    """学習結果を分析"""
    if not os.path.exists(log_dir):
        print(f"ログディレクトリ {log_dir} が見つかりません")
        return
    
    # CSVファイルを探す
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("CSVログファイルが見つかりません")
        return
    
    # 最新のCSVファイルを読み込み
    latest_csv = sorted(csv_files)[-1]
    csv_path = os.path.join(log_dir, latest_csv)
    
    try:
        data = read_csv(csv_path)
        
        print("=== 学習結果分析 ===")
        print(f"総ステップ数: {data['time/total_timesteps'].iloc[-1]:,}")
        print(f"最終エピソード報酬: {data['train/episode_reward_mean'].iloc[-1]:.3f}")
        print(f"最大エピソード報酬: {data['train/episode_reward_mean'].max():.3f}")
        print(f"最小エピソード報酬: {data['train/episode_reward_mean'].min():.3f}")
        
        # 学習曲線をプロット
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(data['time/total_timesteps'], data['train/episode_reward_mean'])
        plt.title('エピソード報酬の推移')
        plt.xlabel('ステップ数')
        plt.ylabel('平均報酬')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(data['time/total_timesteps'], data['train/episode_reward_std'])
        plt.title('報酬の標準偏差')
        plt.xlabel('ステップ数')
        plt.ylabel('標準偏差')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(data['time/total_timesteps'], data['train/entropy_loss'])
        plt.title('エントロピー損失')
        plt.xlabel('ステップ数')
        plt.ylabel('エントロピー損失')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(data['time/total_timesteps'], data['train/policy_loss'])
        plt.title('ポリシー損失')
        plt.xlabel('ステップ数')
        plt.ylabel('ポリシー損失')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("training_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"データの分析に失敗しました: {e}")

def compare_models(model_paths):
    """複数のモデルを比較"""
    print("モデル比較を実行します...")
    
    results = {}
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"モデル {model_path} が見つかりません")
            continue
            
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            
            # テスト環境で評価
            import gymnasium as gym
            import gym_aloha
            
            env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
            rewards = []
            
            for episode in range(10):
                observation, info = env.reset()
                episode_reward = 0
                
                for step in range(1000):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                rewards.append(episode_reward)
            
            env.close()
            
            results[model_path] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards)
            }
            
            print(f"{model_path}: 平均報酬 = {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
            
        except Exception as e:
            print(f"モデル {model_path} の評価に失敗: {e}")
    
    return results

if __name__ == "__main__":
    print("学習結果可視化ツール")
    print("1. エピソード報酬をプロット")
    print("2. Stable-Baselines3ログをプロット")
    print("3. 学習結果を分析")
    print("4. モデル比較")
    
    choice = input("選択してください (1-4): ")
    
    if choice == "1":
        # サンプルデータでデモ
        rewards = np.random.normal(0, 1, 100) + np.linspace(-2, 2, 100)
        plot_episode_rewards(rewards, "サンプル学習進捗", "sample_training.png")
        
    elif choice == "2":
        plot_training_logs()
        
    elif choice == "3":
        analyze_training_results()
        
    elif choice == "4":
        model_paths = input("比較するモデルパスをカンマ区切りで入力: ").split(',')
        compare_models([p.strip() for p in model_paths])
        
    else:
        print("無効な選択です。") 