import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge
import numpy as np
import gymnasium as gym
import gym_aloha

class AlohaSimNode(Node):
    def __init__(self):
        super().__init__('aloha_sim')
        # 環境を作成（obs_type を "pixels_agent_pos" に変更して状態情報を取得）
        self.env = gym.make(
            "gym_aloha/AlohaTransferCube-v0",
            obs_type="pixels_agent_pos",  # qposとqvelにアクセスするために必要
            render_mode="rgb_array"  # 画像レンダリングのみオプション指定
        )
        self.obs, _ = self.env.reset()

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.reward_pub = self.create_publisher(Float32MultiArray, 'reward', 10)

        # Subscriber（外部ノードからアクション受信）
        self.action_sub = self.create_subscription(
            Float32MultiArray, 'action_cmd', self.on_action, 10
        )

        self.bridge = CvBridge()
        self.next_action = None
        # Timer (20Hz) で step() をコール
        self.timer = self.create_timer(0.05, self.step)

    def on_action(self, msg: Float32MultiArray):
        # 外部ノードからのアクションを保存
        self.next_action = np.array(msg.data, dtype=np.float32)

    def step(self):
        # 次のアクション or ランダム
        action = (
            self.next_action
            if self.next_action is not None
            else self.env.action_space.sample()
        )
        # 状態・報酬・終了フラグを取得
        obs, reward, done, truncated, _ = self.env.step(action)

        # Header メッセージを作成
        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        # 1) JointState を publish
        js = JointState()
        js.header = header
        js.name = [f'arm1_joint{i}' for i in range(6)] + [f'arm2_joint{i}' for i in range(6)]
        js.position = list(obs['agent_pos'])  # pixels_agent_posではagent_posキーを使用
        # 速度情報は現在の実装では利用できないため、ゼロで初期化
        js.velocity = [0.0] * len(js.position)
        self.joint_pub.publish(js)

        # 2) カメラ画像を publish
        img = self.env.render()
        ros_img = self.bridge.cv2_to_imgmsg(img, 'rgb8')
        ros_img.header = header
        self.image_pub.publish(ros_img)

        # 3) 報酬を publish
        rm = Float32MultiArray(data=[float(reward)])
        self.reward_pub.publish(rm)

        # 終了 or トランケートされたらリセット
        if done or truncated:
            self.env.reset()


def main(args=None):
    rclpy.init(args=args)
    node = AlohaSimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()