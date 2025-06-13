# ALOHA環境での学習方法

このディレクトリには、ALOHA環境で強化学習を実行するためのスクリプトが含まれています。

## プロジェクト構造

```
lerobot_pj/
├── gym_aloha_ros_humble/              # ROS2パッケージ（メインディレクトリ）
│   ├── gym_aloha_ros_humble/          # Pythonモジュール
│   │   ├── train.py                   # 基本的な学習スクリプト
│   │   ├── train_ppo.py               # PPO学習スクリプト
│   │   ├── visualize_training.py      # 学習結果可視化
│   │   └── simulator_node.py          # ROS2シミュレータノード
│   ├── install_dependencies.sh        # 依存関係インストール
│   ├── best_model/                    # 学習済みモデル保存
│   │   └── best_model.zip
│   ├── logs/                          # 学習ログ
│   │   └── evaluations.npz
│   ├── tensorboard_logs/              # TensorBoardログ
│   │   ├── PPO_1/
│   │   └── PPO_2/
│   ├── test/                          # テストファイル
│   ├── resource/                      # リソースファイル
│   ├── package.xml                    # ROS2パッケージ設定
│   ├── setup.py                       # Pythonパッケージ設定
│   ├── setup.cfg                      # セットアップ設定
│   ├── ppo_aloha_final.zip            # 学習済みPPOモデル
│   └── LEARNING_README.md             # 学習用README
├── src/
│   └── gym-aloha/                     # Gym環境パッケージ
│       ├── gym_aloha/                 # Gym環境モジュール
│       │   ├── env.py                 # メイン環境クラス
│       │   ├── constants.py           # 定数定義
│       │   ├── utils.py               # ユーティリティ関数
│       │   ├── tasks/                 # タスク定義
│       │   │   ├── sim.py
│       │   │   └── sim_end_effector.py
│       │   └── assets/                # 3Dモデルとアセット
│       ├── tests/                     # テストファイル
│       ├── example.py                 # 使用例
│       ├── pyproject.toml             # Pythonプロジェクト設定
│       ├── poetry.lock                # Poetry依存関係ロック
│       └── README.md                  # Gym環境README
├── build/                             # ビルドディレクトリ
│   └── gym_aloha_ros_humble/         # ビルドされたパッケージ
├── install/                           # インストールディレクトリ
│   ├── gym_aloha_ros_humble/         # インストールされたパッケージ
│   ├── setup.bash                     # セットアップスクリプト
│   └── local_setup.bash              # ローカルセットアップスクリプト
├── log/                               # ROS2ログ
│   └── build_*/                       # ビルドログ
└── README.md                          # このファイル
```

## セットアップ

### 1. ROS2環境の準備

```bash
# ROS2 Humbleの環境設定
source /opt/ros/humble/setup.bash

# ワークスペースのビルド
colcon build --packages-select gym_aloha_ros_humble
source install/setup.bash
```

### 2. 依存関係のインストール

```bash
# インストールスクリプトを実行
cd gym_aloha_ros_humble
chmod +x install_dependencies.sh
./install_dependencies.sh
```

または、手動でインストール：

```bash
pip install stable-baselines3[extra] imageio matplotlib tensorboard tqdm pandas seaborn
```

### 3. 環境の確認

```bash
# ALOHA環境が利用可能か確認
python3 -c "import gym_aloha; print('ALOHA環境が利用可能です')"
```

## 学習の実行

### 1. 基本的な学習（ランダムエージェント・Q-Learning）

```bash
cd gym_aloha_ros_humble
python3 gym_aloha_ros_humble/train.py
```

選択肢：
- **1**: ランダムエージェント（100エピソード）
- **2**: Q-Learningエージェント（200エピソード）
- **3**: デモ動画の録画

### 2. PPO（Proximal Policy Optimization）学習

```bash
cd gym_aloha_ros_humble
python3 gym_aloha_ros_humble/train_ppo.py
```

選択肢：
- **1**: PPOで学習（100万ステップ）
- **2**: 学習済みモデルのテスト
- **3**: 学習済みモデルでデモ録画

### 3. 学習結果の可視化

```bash
cd gym_aloha_ros_humble
python3 gym_aloha_ros_humble/visualize_training.py
```

選択肢：
- **1**: エピソード報酬のプロット
- **2**: Stable-Baselines3ログのプロット
- **3**: 学習結果の詳細分析
- **4**: 複数モデルの比較

## 学習パラメータの調整

### PPO学習のパラメータ

`train_ppo.py`の以下のパラメータを調整できます：

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,    # 学習率
    n_steps=2048,          # バッチサイズ
    batch_size=64,         # ミニバッチサイズ
    n_epochs=10,           # エポック数
    gamma=0.99,            # 割引率
    gae_lambda=0.95,       # GAEパラメータ
    clip_range=0.2,        # クリップ範囲
    tensorboard_log="./tensorboard_logs/"
)
```

### 学習ステップ数の調整

```python
model.learn(
    total_timesteps=1000000,  # 学習ステップ数
    callback=eval_callback,
    progress_bar=True
)
```

## 学習の監視

### TensorBoardでの監視

```bash
# 学習中にTensorBoardを起動
cd gym_aloha_ros_humble
tensorboard --logdir ./tensorboard_logs/

# ブラウザで http://localhost:6006 にアクセス
```

### ログファイルの確認

学習中に以下のディレクトリにログが保存されます：
- `./logs/`: 学習ログ（CSV形式）
- `./best_model/`: 最良モデル
- `./tensorboard_logs/`: TensorBoardログ

## 学習済みモデルの使用

### モデルの保存と読み込み

```python
# モデルの保存
model.save("my_trained_model")

# モデルの読み込み
from stable_baselines3 import PPO
model = PPO.load("my_trained_model")
```

### 学習済みモデルでの推論

```python
# 環境の作成
env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")

# 推論
observation, info = env.reset()
for step in range(1000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
```

## ROS2との連携

### シミュレータノードの実行

```bash
# ROS2ノードとして実行
ros2 run gym_aloha_ros_humble simulator_node
```

### 学習済みモデルをROSノードで使用

学習済みモデルをROSノードで使用する場合：

1. モデルを読み込み
2. `simulator_node.py`の`on_action`メソッドで推論を実行
3. 推論結果をアクションとして送信

```python
# simulator_node.pyでの使用例
def on_action(self, msg: Float32MultiArray):
    # 学習済みモデルで推論
    action, _ = self.model.predict(self.obs, deterministic=True)
    self.next_action = action
```

## トラブルシューティング

### よくある問題

1. **ImportError: No module named 'gym_aloha'**
   ```bash
   # ワークスペースをビルドしてソース
   colcon build --packages-select gym_aloha_ros_humble
   source install/setup.bash
   ```

2. **ROS2パッケージが見つからない**
   ```bash
   # パッケージのビルドとソース
   colcon build --packages-select gym_aloha_ros_humble
   source install/setup.bash
   ```

3. **CUDAエラー**
   ```bash
   # CPUのみで実行
   export CUDA_VISIBLE_DEVICES=""
   ```

4. **メモリ不足**
   - バッチサイズを小さくする
   - 学習ステップ数を減らす

### パフォーマンスの最適化

1. **GPU使用**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **並列環境**
   ```python
   # 複数環境で並列学習
   env = DummyVecEnv([make_env() for _ in range(4)])
   ```

## 開発環境のセットアップ

### 新しい機能の追加

1. `gym_aloha_ros_humble/gym_aloha_ros_humble/`にPythonファイルを追加
2. `package.xml`と`setup.py`を更新
3. ワークスペースをビルド

```bash
colcon build --packages-select gym_aloha_ros_humble
source install/setup.bash
```

### デバッグ

```bash
# ROS2ログの確認
ros2 log list
ros2 log show /gym_aloha_ros_humble

# ノードの状態確認
ros2 node list
ros2 node info /simulator_node
```

## 参考資料

- [Stable-Baselines3 ドキュメント](https://stable-baselines3.readthedocs.io/)
- [Gymnasium ドキュメント](https://gymnasium.farama.org/)
- [ROS2 ドキュメント](https://docs.ros.org/en/humble/)
- [ALOHA プロジェクト](https://github.com/tonyzhaozh/ALOHA) 