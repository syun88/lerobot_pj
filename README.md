# ALOHA ROS2 シミュレーション環境

このプロジェクトは、ALOHA（双腕ロボット）のシミュレーション環境をROS2（Humble）で統合したものです。gym-aloha環境をROS2ノードとして実行し、ロボットの状態情報やカメラ画像をROS2トピックとして配信します。

## 概要

ALOHAは、双腕ロボット（ViperX 300s）を使用したマニピュレーションタスクのためのシミュレーション環境です。以下のタスクが利用可能です：

- **TransferCubeTask**: 右腕で赤いキューブを拾い上げ、左腕のグリッパーに移動させるタスク
- **InsertionTask**: 左右の腕でソケットとペグをそれぞれ拾い上げ、空中で挿入するタスク

## プロジェクト構造

```
lerobot_pj/
├── src/
│   ├── gym_aloha_ros_humble/     # ROS2パッケージ
│   └── gym-aloha/                # gym-aloha環境
├── build/                        # ビルドディレクトリ
├── install/                      # インストールディレクトリ
└── log/                         # ログディレクトリ
```

## 前提条件

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- MuJoCo 2.3.7以上

## セットアップ

### 1. 依存関係のインストール

```bash

# 必要なパッケージのインストール
pip install gym-aloha
pip install opencv-python
```

### 2. ROS2ワークスペースのビルド

```bash
# ワークスペースのルートディレクトリで
colcon build
source install/setup.bash
```

## 使用方法

### シミュレーターノードの起動

```bash
ros2 run gym_aloha_ros_humble simulator_node
```

このコマンドにより、以下のROS2トピックが利用可能になります：

- `/joint_states` (sensor_msgs/JointState): ロボットの関節状態
- `/camera/image_raw` (sensor_msgs/Image): カメラ画像
- `/reward` (std_msgs/Float32MultiArray): 報酬情報
- `/action_cmd` (std_msgs/Float32MultiArray): アクションコマンド（購読）

### 可視化ツールの使用

#### RQTを使用した画像表示

```bash
rqt
```

RQTを起動し、Plugins > Visualization > Image View を選択してカメラ画像を表示できます。

#### RVizを使用した可視化

```bash
rviz2
```

RVizで以下の設定を行ってください：
- Fixed Frame: `world`
- Add > By topic > `/joint_states` > RobotModel
(開発中)
- Add > By topic > `/camera/image_raw` > Image

### アクションの送信

外部ノードからアクションを送信する場合：

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class ActionPublisher(Node):
    def __init__(self):
        super().__init__('action_publisher')
        self.publisher = self.create_publisher(Float32MultiArray, 'action_cmd', 10)
        
    def publish_action(self, action):
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.publisher.publish(msg)

# 使用例
rclpy.init()
node = ActionPublisher()
action = np.random.random(14)  # 14次元のアクション
node.publish_action(action)
```

## アクション空間

アクションは14次元のベクトルで構成されています：

- 左腕の関節位置（6次元）
- 左グリッパーの位置（1次元、0:閉じる、1:開く）
- 右腕の関節位置（6次元）
- 右グリッパーの位置（1次元、0:閉じる、1:開く）

## 観測空間

観測は以下の情報を含みます：

- `qpos`: ロボットの関節位置とグリッパー位置
- `qvel`: ロボットの関節速度とグリッパー速度
- `images`: 複数のカメラアングルからの画像
- `env_state`: 環境の状態情報（オブジェクトの位置など）

## 報酬システム

### TransferCubeTask
- 1点: 右グリッパーでキューブを保持
- 2点: 右グリッパーでキューブを持ち上げ
- 3点: 左グリッパーにキューブを移動
- 4点: テーブルに触れずに成功

### InsertionTask
- 1点: ペグとソケットの両方にグリッパーで接触
- 2点: 両方を落とさずに把持
- 3点: ペグがソケットに整列して接触
- 4点: ペグをソケットに挿入成功

## トラブルシューティング

### GPUレンダリングの問題

MuJoCoがGPUレンダリングに問題がある場合、以下の設定を確認してください：

```python
import os
os.environ['MUJOCO_GL'] = 'egl'
```

### カメラ画像が表示されない場合

RQTで画像が表示されない場合は、以下を確認してください：
- シミュレーターノードが正常に起動しているか
- トピック名が正しいか（`/camera/image_raw`）
- 画像エンコーディングが`rgb8`になっているか

## 開発向け情報

###　タスクの追加

新しいタスクを追加する場合は、`src/gym-aloha/gym_aloha/tasks/`ディレクトリに新しいタスククラスを作成し、`simulator_node.py`で環境を変更してください。
gazebo　jointの設定など

