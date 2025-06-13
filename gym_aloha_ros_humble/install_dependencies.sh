#!/bin/bash

echo "ALOHA学習用の依存関係をインストールします..."

# 基本的な学習ライブラリ
pip install stable-baselines3[extra]
pip install imageio
pip install matplotlib
pip install tensorboard

# 追加の便利なライブラリ
pip install tqdm
pip install pandas
pip install seaborn

echo "依存関係のインストールが完了しました！"

echo ""
echo "使用方法:"
echo "1. 基本的な学習: python3 gym_aloha_ros_humble/train.py"
echo "2. PPO学習: python3 gym_aloha_ros_humble/train_ppo.py"
echo "3. ROSノード実行: ros2 run gym_aloha_ros_humble simulator_node" 