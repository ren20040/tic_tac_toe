#!/usr/bin/env python3
#6颗棋,优化了动作
import rospy
import math
import numpy as np
import time
# 导入argparse并正确使用
import argparse
import socket
import pickle
from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from kuavo_msgs.srv import twoArmHandPoseCmdSrv
from kuavo_msgs.msg import twoArmHandPoseCmd, ikSolveParam
from kuavo_msgs.msg import robotHeadMotionData
from kuavo_msgs.msg import robotHandPosition

####################################################################

# ====================== 全局可配置参数（仅修改这里即可）======================
# 棋子位置配置参数（左右两个棋子区域；每区3个棋子排成一列）
# 说明：
# - 左手下棋：从左侧棋子区域取（LEFT_*）
# - 右手下棋：从右侧棋子区域取（RIGHT_*）
# - 每个区域三个棋子：slot=1,2,3；沿 X 方向等间距排列（间距共用 PIECE_X_SPACING）
#
# 量好后的坐标请直接改下面 6 个常量：
LEFT_PIECE_START_X = 0.35     # 左侧棋子区域起始X（slot=1）
LEFT_PIECE_START_Y = 0.27    # 左侧棋子区域起始Y（slot=1）
LEFT_PIECE_Z = 0.02           # 左侧棋子区域Z

RIGHT_PIECE_START_X = 0.33    # 右侧棋子区域起始X（slot=1）
RIGHT_PIECE_START_Y = -0.27  # 右侧棋子区域起始Y（slot=1）
RIGHT_PIECE_Z = 0.02         # 右侧棋子区域Z

PIECE_X_SPACING = 0.12       # 同一区域内两个棋子的X轴间距（8cm）
PIECE_SLOTS_PER_REGION = 3    # 每个区域棋子数量（修改为3）

# 棋盘位置配置参数
BOARD_START_X = 0.59   # 棋盘起始X坐标（第一行）
BOARD_START_Y = 0.1  # 棋盘起始Y坐标（第一列）
BOARD_X_SPACING = 0.10 # 棋盘行间距（X轴）
BOARD_Y_SPACING = 0.10 # 棋盘列间距（Y轴）
BOARD_Z = 0.10       # 棋盘统一高度
BOARD_ROWS = 3         # 棋盘行数
BOARD_COLS = 3         # 棋盘列数

# TCP服务端默认配置（可通过命令行覆盖）
HOST = "0.0.0.0"
PORT = 8888
BUFFER_SIZE = 1024
# ============================================================================

def _build_piece_regions():
    """生成左右两个棋子区域的位置表：{arm_side: {slot: (x,y,z)}}"""
    regions = {
        'left':  (LEFT_PIECE_START_X,  LEFT_PIECE_START_Y,  LEFT_PIECE_Z),
        'right': (RIGHT_PIECE_START_X, RIGHT_PIECE_START_Y, RIGHT_PIECE_Z),
    }
    out = {'left': {}, 'right': {}}
    for side, (sx, sy, sz) in regions.items():
        for slot in range(1, PIECE_SLOTS_PER_REGION + 1):
            x = sx + (slot - 1) * PIECE_X_SPACING
            out[side][slot] = (round(x, 3), round(sy, 3), round(sz, 3))
    return out

PIECE_REGIONS = _build_piece_regions()

def resolve_piece_position(arm_side, slot_num):
    """
    根据 arm_side 和 slot_num 选择棋子区域和槽位。

    参数:
    arm_side: 'left' 或 'right'，指定机械臂侧
    slot_num: 槽位编号，范围 1-3

    返回:
    (slot, pos): (槽位编号, 三维坐标)
    """
    if arm_side not in ('left', 'right'):
        raise ValueError(f"arm_side 非法: {arm_side}")

    if not (1 <= slot_num <= 3):
        raise ValueError(f"slot_num 非法(仅支持 1-3): {slot_num}")

    pos = PIECE_REGIONS[arm_side].get(slot_num)
    if pos is None:
        raise ValueError(f"未配置棋子位置: arm_side={arm_side}, slot={slot_num}")
    return slot_num, pos

# 遍历生成棋盘位置字典
BOARD_POSITIONS = {}
for row in range(1, BOARD_ROWS + 1):
    x = BOARD_START_X - (row - 1) * BOARD_X_SPACING
    for col in range(1, BOARD_COLS + 1):
        y = BOARD_START_Y - (col - 1) * BOARD_Y_SPACING
        BOARD_POSITIONS[(row, col)] = (round(x, 2), round(y, 2), BOARD_Z)

# 左臂优先执行的棋盘坐标（历史兼容/备用）
# 说明：实际左右手选择以 select_arm_side() 中的“按列强制规则”为准：
# - 最左边一列：强制左手
# - 最右边两列：强制右手
LEFT_ARM_BOARD_POSITIONS = {
    (1, 1), (2, 1), (3, 1),  # 第一列全部（col=1，左侧）
}


# 自定义ik参数
use_custom_ik_param = True
joint_angles_as_q0 = False
ik_solve_param = ikSolveParam()
ik_solve_param.major_optimality_tol = 1e-5
ik_solve_param.major_feasibility_tol = 1e-5
ik_solve_param.minor_feasibility_tol = 1e-5
ik_solve_param.major_iterations_limit = 500
ik_solve_param.oritation_constraint_tol= 1e-5
ik_solve_param.pos_constraint_tol = 1e-5
ik_solve_param.pos_cost_weight = 1.0

# 灵巧手参数
hand_open_value = 0
hand_hold_value = 85
hand_full_close_value = 100
hand_default_delay = 0.5

# ====================== 关键姿态（角度制）======================
# 说明：
# - INIT：双臂初始安全姿态（原脚本多处用到 20/0/0/-30）
# - READY：双臂准备姿态（你希望“非工作手”始终保持的姿态）

trajectory_times = [3.0, 5.0]

mid_pose = [-5.0, 70.0, 0.10, -20.0, 0.0, 0.0, 0.0,
            -5.0, -70.0, 0.10, -20.0, 0.0, 0.0, 0.0]




INIT_BOTH_ARM_JOINT_DEG = [20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
                           20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]


final_pose = INIT_BOTH_ARM_JOINT_DEG

trajectory_values = mid_pose +  final_pose

READY_LEFT_ARM_JOINT_DEG =  [0.0,  60.0, 0.0, -90.0, 0.0, 0.0, 0.0]
READY_RIGHT_ARM_JOINT_DEG = [0.0, -60.0, 0.0, -90.0, 0.0, 0.0, 0.0]

READY_OPEN_LEFT_ARM_JOINT_DEG =  [20.0,  60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
READY_OPEN_RIGHT_ARM_JOINT_DEG = [20.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 头部控制
def set_head_target(yaw, pitch):
    pub_head_pose = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
    rospy.sleep(0.5)
    head_target_msg = robotHeadMotionData()
    head_target_msg.joint_data = [yaw, pitch]
    pub_head_pose.publish(head_target_msg)
    rospy.loginfo(f"Published head target: yaw={yaw}, pitch={pitch}")

########################### 灵巧手控制逻辑 #########################################
def publish_controlEndHand(hand_traj):
    """
    控制机器人的手部动作。

    参数:
    hand_traj (list): 包含左手和右手位置的列表，前6个元素为左手，后6个元素为右手。

    返回:
    bool: 发布结果，成功返回True，失败返回False。
    """
    try:
        pub = rospy.Publisher('control_robot_hand_position', robotHandPosition, queue_size=10)
        rospy.sleep(0.5)
        msg = robotHandPosition()
        msg.left_hand_position = hand_traj[0:6]
        msg.right_hand_position = hand_traj[6:]

        rate = rospy.Rate(10)
        while pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("等待订阅者连接...")
            rate.sleep()

        pub.publish(msg)
        rospy.loginfo(f"发布手部位置: 左手={msg.left_hand_position}, 右手={msg.right_hand_position}")
        return True

    except rospy.ServiceException as e:
        rospy.logerr(f"controlEndHand 服务调用失败: {e}")
        return False

def grip_all(hand='right', grip_value=100, delay=0.5):
    """
    同时握紧所有手指

    参数:
    hand (str): 'left'左手, 'right'右手, 'both'双手
    grip_value (float): 0-100，100为完全握紧
    """
    left_hand = [0, 0, 0, 0, 0, 0]
    right_hand = [0, 0, 0, 0, 0, 0]

    if hand == 'left' or hand == 'both':
        left_hand = [grip_value] * 6

    if hand == 'right' or hand == 'both':
        right_hand = [grip_value] * 6

    hand_traj = left_hand + right_hand

    rospy.loginfo(f"【同时握紧】{hand}手，目标值={grip_value}")
    publish_controlEndHand(hand_traj)
    rospy.sleep(delay)

def sequential_grip(hand='right', grip_value=100, delay=0.5):
    """
    依次握紧手指，从小拇指开始到大拇指

    参数:
    hand (str): 控制哪只手，'left'表示左手，'right'表示右手，'both'表示双手
    grip_value (float): 握紧的目标值（0-100），0表示张开，100表示完全握紧
    delay (float): 每个手指动作之间的延迟时间（秒）

    手指索引说明:
    - 索引 0, 1: 大拇指的两个自由度
    - 索引 2: 食指
    - 索引 3: 中指
    - 索引 4: 无名指
    - 索引 5: 小拇指

    握紧顺序: 小拇指(5) -> 无名指(4) -> 中指(3) -> 食指(2) -> 大拇指(1,0)
    """

    left_hand = [0, 0, 0, 0, 0, 0]
    right_hand = [0, 0, 0, 0, 0, 0]

    finger_sequence = [5, 4, 3, 2, 1, 0]
    finger_names = ['小拇指', '无名指', '中指', '食指', '大拇指第二关节', '大拇指第一关节']

    rospy.loginfo(f"开始依次握紧{hand}手，从小拇指到大拇指...")
    rospy.loginfo(f"握紧目标值: {grip_value}, 每个手指延迟: {delay}秒")

    for i, finger_index in enumerate(finger_sequence):
        if hand == 'left' or hand == 'both':
            left_hand[finger_index] = grip_value

        if hand == 'right' or hand == 'both':
            right_hand[finger_index] = grip_value

        hand_traj = left_hand + right_hand

        rospy.loginfo(f"步骤 {i+1}/{len(finger_sequence)}: 握紧 {finger_names[i]} (索引{finger_index})")
        publish_controlEndHand(hand_traj)
        rospy.sleep(delay)

    rospy.loginfo("依次握紧完成！")

def sequential_release(hand='right', delay=0.5):
    """
    依次松开手指，从大拇指开始到小拇指（与握紧相反的顺序）

    参数:
    hand (str): 控制哪只手，'left'表示左手，'right'表示右手，'both'表示双手
    delay (float): 每个手指动作之间的延迟时间（秒）
    """

    grip_value = 100
    left_hand = [grip_value] * 6
    right_hand = [grip_value] * 6

    finger_sequence = [0, 1, 2, 3, 4, 5]
    finger_names = ['大拇指第一关节', '大拇指第二关节', '食指', '中指', '无名指', '小拇指']

    rospy.loginfo(f"开始依次松开{hand}手，从大拇指到小拇指...")

    for i, finger_index in enumerate(finger_sequence):
        if hand == 'left' or hand == 'both':
            left_hand[finger_index] = 0

        if hand == 'right' or hand == 'both':
            right_hand[finger_index] = 0

        hand_traj = left_hand + right_hand

        rospy.loginfo(f"步骤 {i+1}/{len(finger_sequence)}: 松开 {finger_names[i]} (索引{finger_index})")
        publish_controlEndHand(hand_traj)
        rospy.sleep(delay)

    rospy.loginfo("依次松开完成！")

######################## ik求解部分 ############################################
def get_parameter(param_name):
    try:
        param_value = rospy.get_param(param_name)
        rospy.loginfo(f"参数 {param_name} 的值为: {param_value}")
        return param_value
    except rospy.ROSException:
        rospy.logerr(f"参数 {param_name} 不存在！程序退出。")
        rospy.signal_shutdown("参数获取失败")
        return None

def call_ik_srv(eef_pose_msg):
    rospy.wait_for_service('/ik/two_arm_hand_pose_cmd_srv')
    try:
        ik_srv = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
        res = ik_srv(eef_pose_msg)
        return res
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
        return False, []

def set_arm_control_mode(mode):
    arm_traj_change_mode_client = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)
    request = changeArmCtrlModeRequest()
    request.control_mode = mode
    response = arm_traj_change_mode_client(request)
    if response.result:
        rospy.loginfo(f"Successfully changed arm control mode to {mode}: {response.message}")
    else:
        rospy.logwarn(f"Failed to change arm control mode to {mode}: {response.message}")

def publish_arm_target_poses(times, values):
    pub = rospy.Publisher('kuavo_arm_target_poses', armTargetPoses, queue_size=10)
    rospy.sleep(0.5)
    msg = armTargetPoses()
    msg.times = times
    msg.values = values
    rospy.loginfo("发布手臂目标姿态到话题 'kuavo_arm_target_poses'")

    rate = rospy.Rate(10)
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.loginfo("等待订阅者连接...")
        rate.sleep()

    pub.publish(msg)
    rospy.loginfo("手臂目标姿态已发布。")


def _compose_two_arm_deg(left7, right7):
    if len(left7) != 7 or len(right7) != 7:
        raise ValueError("left7/right7 必须是7维关节角（角度制）")
    return list(left7) + list(right7)


def _deg7_to_rad7(deg7):
    return np.array([math.radians(d) for d in deg7], dtype=float)

class Quaternion:
    def __init__(self):
        self.w = 0
        self.x = 0
        self.y = 0
        self.z = 0

def euler_to_rotation_matrix(yaw_adaptive=0, pitch_adaptive=0, roll_adaptive=0,
                            yaw_manual=0, pitch_manual=0, roll_manual=0):
    cy, sy = np.cos(yaw_adaptive), np.sin(yaw_adaptive)
    cp, sp = np.cos(pitch_adaptive), np.sin(pitch_adaptive)

    R = np.array([
        [cy * cp,   -sy,        cy * sp],
        [sy * cp,    cy,        sy * sp],
        [-sp,        0,         cp     ]
    ])

    if yaw_manual or pitch_manual or roll_manual:
        R_manual = np.array([[1,0,0],[0,1,0],[0,0,1]])
        if abs(yaw_manual) > 0.01:
            c, s = np.cos(yaw_manual), np.sin(yaw_manual)
            R_manual = np.array([[c, -s, 0],[s, c, 0],[0,0,1]]) @ R_manual
        if abs(pitch_manual) > 0.01:
            c, s = np.cos(pitch_manual), np.sin(pitch_manual)
            R_manual = np.array([[c,0,s],[0,1,0],[-s,0,c]]) @ R_manual
        if abs(roll_manual) > 0.01:
            c, s = np.cos(roll_manual), np.sin(roll_manual)
            R_manual = np.array([[1,0,0],[0,c,-s],[0,s,c]]) @ R_manual
        return R @ R_manual
    else :
        return R

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    q = Quaternion()
    if trace > 0:
        q.w = math.sqrt(trace + 1.0) / 2
        q.x = (R[2, 1] - R[1, 2]) / (4 * q.w)
        q.y = (R[0, 2] - R[2, 0]) / (4 * q.w)
        q.z = (R[1, 0] - R[0, 1]) / (4 * q.w)
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        t = np.zeros(4)
        t[i] = math.sqrt(R[i,i] - R[j,j] - R[k,k] +1)/2
        t[j] = (R[i,j]+R[j,i])/(4*t[i])
        t[k] = (R[i,k]+R[k,i])/(4*t[i])
        t[3] = (R[k,j]-R[j,k])/(4*t[i])
        q.x, q.y, q.z, q.w = t
    norm = math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
    if norm>0:
        q.w /= norm
        q.x /= norm
        q.y /= norm
        q.z /= norm
    return q

def euler_to_quaternion_via_matrix(yaw_adaptive=0, pitch_adaptive=0, roll_adaptive=0,
                                    yaw_manual=0, pitch_manual=0, roll_manual=0):
    R = euler_to_rotation_matrix(yaw_adaptive, pitch_adaptive, roll_adaptive,
                                yaw_manual, pitch_manual, roll_manual)
    return rotation_matrix_to_quaternion(R)

########################### 核心：分阶段自适应准备姿态 #########################################
"""
def set_arm_ready_pose(arm_side):
    rospy.loginfo(f"机械臂开始执行{arm_side}手准备姿态...")
    # publish_arm_target_poses([1.5], [20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
    #                                20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0])
    # time.sleep(1.5)
    # 关键修改：无论哪只手进入准备姿态，另一只手都保持 READY（不再被拉回初始/下垂位）
    if arm_side == 'left':
        publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_OPEN_LEFT_ARM_JOINT_DEG, READY_RIGHT_ARM_JOINT_DEG))
    else:
        publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, READY_OPEN_RIGHT_ARM_JOINT_DEG))
    time.sleep(1.5)
    if arm_side == 'left':
        publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, READY_RIGHT_ARM_JOINT_DEG))
    else:
        publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, READY_RIGHT_ARM_JOINT_DEG))
    time.sleep(1.5)
    rospy.loginfo(f"{arm_side}手准备姿态执行完成，待命抓取...")
"""
def set_both_arms_ready_pose():
    rospy.loginfo("机械臂开始执行双手准备姿态...")
    publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_OPEN_LEFT_ARM_JOINT_DEG, (20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0)))
    time.sleep(1.5)
    publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, (20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0)))
    time.sleep(1.5)
    publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, READY_OPEN_RIGHT_ARM_JOINT_DEG))
    time.sleep(1.5)
    publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, READY_RIGHT_ARM_JOINT_DEG))
    time.sleep(1.5)
    rospy.loginfo("双手准备姿态执行完成，待命抓取...")


########################### 机械臂控制核心函数 #########################################
def init_robot(arm_side):

    rospy.loginfo("开始初始化机器人...")
    set_head_target(0, 25)
    time.sleep(2)
    set_arm_control_mode(2)

    # 灵巧手张开复位
    grip_all(hand='both', grip_value=hand_open_value, delay=1.0)

    # 先明确发布一次“初始安全姿态”，作为原路回放的起点
    publish_arm_target_poses([1.5], INIT_BOTH_ARM_JOINT_DEG)
    time.sleep(1.6)

    set_both_arms_ready_pose()
    return publish_controlEndHand


def move_arm_to_position(control_leju_claw, target_pos, arm_side, custom_quat=None):
    robot_version = get_parameter('robot_version')
    def start_with_version(version_number:int, series:int):
        MMMMN_MASK = 100000
        return (version_number % MMMMN_MASK) == series

    if start_with_version(robot_version, 45) or start_with_version(robot_version, 49):
        robot_zero_x = -0.0173
        robot_zero_y = -0.2927
        robot_zero_z = -0.2837
    elif start_with_version(robot_version, 42):
        robot_zero_x = -0.0175
        robot_zero_y = -0.25886
        robot_zero_z = -0.20115
    else :
        rospy.logerr("机器人版本号错误, 仅支持 42 45 49 系列")
        return False

    eef_pose_msg = twoArmHandPoseCmd()
    eef_pose_msg.ik_param = ik_solve_param
    eef_pose_msg.use_custom_ik_param = use_custom_ik_param
    eef_pose_msg.joint_angles_as_q0 = joint_angles_as_q0
    eef_pose_msg.hand_poses.left_pose.joint_angles = np.zeros(7)
    eef_pose_msg.hand_poses.right_pose.joint_angles = np.zeros(7)

    set_x, set_y, set_z = target_pos
    if custom_quat is not None:
        quat = custom_quat
    else:
        if arm_side == 'left':
            quat = euler_to_quaternion_via_matrix(-math.pi/6, -math.pi/2, 0, -math.pi/2, math.pi/8, 0)
        else:
            quat = euler_to_quaternion_via_matrix(math.pi/6, -math.pi/2, 0, math.pi/2, math.pi/8, 0)


    if arm_side == 'left':
        eef_pose_msg.hand_poses.left_pose.pos_xyz = np.array([set_x, set_y, set_z])
        eef_pose_msg.hand_poses.left_pose.quat_xyzw = [quat.x, quat.y, quat.z, quat.w]
        eef_pose_msg.hand_poses.right_pose.pos_xyz = np.array([robot_zero_x, robot_zero_y, robot_zero_z + 0.1])
        eef_pose_msg.hand_poses.right_pose.quat_xyzw = [0.0,0.0,0.0,1.0]
    else:
        eef_pose_msg.hand_poses.right_pose.pos_xyz = np.array([set_x, set_y, set_z])
        eef_pose_msg.hand_poses.right_pose.quat_xyzw = [quat.x, quat.y, quat.z, quat.w]
        eef_pose_msg.hand_poses.left_pose.pos_xyz = np.array([robot_zero_x, -1*robot_zero_y, robot_zero_z + 0.1])
        eef_pose_msg.hand_poses.left_pose.quat_xyzw = [0.0,0.0,0.0,1.0]

    rospy.loginfo(f"求解{arm_side}手臂逆解，目标位置: {target_pos}")
    res = call_ik_srv(eef_pose_msg)
    if not res.success:
        rospy.logerr("IK逆解失败！")
        return False

    l_pos_error = np.linalg.norm(res.hand_poses.left_pose.pos_xyz - eef_pose_msg.hand_poses.left_pose.pos_xyz)
    r_pos_error = np.linalg.norm(res.hand_poses.right_pose.pos_xyz - eef_pose_msg.hand_poses.right_pose.pos_xyz)
    rospy.loginfo(f"IK耗时: {res.time_cost:.2f}ms, 误差:左{l_pos_error*1e3:.2f}mm 右{r_pos_error*1e3:.2f}mm")

    # 关键修改：非工作手固定保持 READY（不再被拉回 INIT/下垂位）
    passive_left_rad = _deg7_to_rad7(READY_LEFT_ARM_JOINT_DEG)
    passive_right_rad = _deg7_to_rad7(READY_RIGHT_ARM_JOINT_DEG)
    if arm_side == 'left':
        joint_end_angles = np.concatenate([res.hand_poses.left_pose.joint_angles, passive_right_rad])
    else:
        joint_end_angles = np.concatenate([passive_left_rad, res.hand_poses.right_pose.joint_angles])

    degrees_list = [math.degrees(rad) for rad in joint_end_angles]
    publish_arm_target_poses([3], degrees_list)
    time.sleep(4)
    return True

def grasp_piece(control_leju_claw, slot_num, arm_side):
    try:
        # 直接使用 slot_num，不再需要 resolve_piece_position
        pos = PIECE_REGIONS[arm_side].get(slot_num)
        if pos is None:
            raise ValueError(f"未配置棋子位置: arm_side={arm_side}, slot={slot_num}")
        piece_pos = pos
    except ValueError as e:
        rospy.logerr(str(e))
        return False

    above_piece_pos = (piece_pos[0], piece_pos[1], piece_pos[2] + 0.05)
    if not move_arm_to_position(control_leju_claw, above_piece_pos, arm_side):
        return False

    grip_all(hand=arm_side, grip_value=hand_open_value, delay=1.0)

    if not move_arm_to_position(control_leju_claw, piece_pos, arm_side):
        return False
    grip_all(hand=arm_side, grip_value=hand_hold_value, delay=1.5)
    rospy.loginfo(f"已抓取棋子：arm_side={arm_side}, slot={slot_num}, pos={piece_pos}")

    lift_pos = (piece_pos[0], piece_pos[1], piece_pos[2] + 0.25)
    if not move_arm_to_position(control_leju_claw, lift_pos, arm_side):
        return False
    return True

def place_piece(control_leju_claw, board_pos, arm_side):
    if board_pos not in BOARD_POSITIONS:
        rospy.logerr(f"无效棋盘位置: {board_pos}")
        return False
    board_row, board_col = board_pos
    target_board_pos = BOARD_POSITIONS[board_pos]

    #if board_col == 1:
    #    place_quat = euler_to_quaternion_via_matrix(math.pi/6, -math.pi/2, 0, math.pi/2, math.pi/8, 0)

    if arm_side == 'left':
        place_quat = euler_to_quaternion_via_matrix(-math.pi/6, -math.pi/2, 0, -math.pi/2, math.pi/8, 0)
    else:
        place_quat = euler_to_quaternion_via_matrix(math.pi/6, -math.pi/2, 0, math.pi/2, math.pi/8, 0)


    above_board_pos = (target_board_pos[0], target_board_pos[1], target_board_pos[2] + 0.10)
    if not move_arm_to_position(control_leju_claw, above_board_pos, arm_side, place_quat):
        return False
    if not move_arm_to_position(control_leju_claw, target_board_pos, arm_side, place_quat):
        return False

    grip_all(hand=arm_side, grip_value=hand_open_value, delay=1.5)
    rospy.loginfo(f"已放置棋子到 {board_pos}")

    lift_pos = (target_board_pos[0], target_board_pos[1], target_board_pos[2] + 0.35)
    if not move_arm_to_position(control_leju_claw, lift_pos, arm_side, place_quat):
        return False
    if arm_side == 'left':
        lift2_pos = (target_board_pos[0], target_board_pos[1] + 0.1, target_board_pos[2] + 0.35)
    else:
        lift2_pos = (target_board_pos[0], target_board_pos[1] - 0.1, target_board_pos[2] + 0.35)
    if not move_arm_to_position(control_leju_claw, lift2_pos, arm_side, place_quat):
        return False
    return True

########################### 机械臂选择逻辑 #########################################
def select_arm_side(board_row, board_col, default_arm_side):

    # 强制规则（按“列”判断），满足你的需求：
    # - 落子到最左边一列：用左手
    # - 落子到最右边两列：用右手
    # 其余列（如果棋盘扩展到 >3 列）：保持默认策略不变
    if board_col == 1:
        return 'left'

    # 兼容不同棋盘列数：最后两列强制右手
    # 例如 3x3：col=2/3 都是右手；8x8：col=7/8 强制右手
    if board_col >= max(1, BOARD_COLS - 1):
        return 'right'

    # 兜底：保留历史配置/默认手臂策略
    if (board_row, board_col) in LEFT_ARM_BOARD_POSITIONS:
        return 'left'
    return default_arm_side

########################### 任务执行函数（修复未定义args问题） #########################################
def run_chess_task(control_leju_claw, slot_num, board_row, board_col, arm_side='right', cost_weight=0.0):
    """封装机械臂执行逻辑，接收网络传输的参数"""


    # 新增双层校验
    if not (1 <= slot_num <= 3) or not (1 <= board_row <= 3 and 1 <= board_col <= 3):
        rospy.logerr(f"任务参数非法，终止执行：slot_num={slot_num}, pos=({board_row},{board_col})")
        return False

    # 修复：使用传入的cost_weight参数，不再依赖未定义的args
    if cost_weight != 0.0:
        ik_solve_param.pos_cost_weight = cost_weight
    try:
        # control_leju_claw = init_robot(arm_side)
        board_position = (board_row, board_col)
        rospy.loginfo(f"执行任务：抓取{arm_side}侧slot={slot_num}棋子 → 放置到{board_position}")

        if not grasp_piece(control_leju_claw, slot_num, arm_side):
            rospy.logerr("抓取失败")
            return False
        if not place_piece(control_leju_claw, board_position, arm_side):
            rospy.logerr("放置失败")
            return False

    ########################################## 运动控制 后续处理 #########################################
        # 手臂复位：两手都回到 READY，确保“非工作手”持续保持准备姿态
        publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, READY_RIGHT_ARM_JOINT_DEG))

        
        # 3. 灵巧手保持张开
        grip_all(hand=arm_side, grip_value=hand_open_value, delay=1.0)

        rospy.loginfo("任务执行完成！")
        return True
    except Exception as e:
        rospy.logerr(f"任务异常：{str(e)}")
        return False

########################### 命令行参数解析函数 #########################################
def parse_args():
    """创建命令行参数解析器，支持灵活配置程序参数"""
    parser = argparse.ArgumentParser(description="机械臂象棋操控TCP服务端")
    # TCP通信参数
    parser.add_argument('--host', type=str, default=HOST, help=f"TCP服务端监听地址，默认：{HOST}")
    parser.add_argument('--port', type=int, default=PORT, help=f"TCP服务端监听端口，默认：{PORT}")
    # 机械臂配置参数
    parser.add_argument('--arm-side', type=str, default='right', choices=['left', 'right'], help="指定工作机械臂，默认：right")
    parser.add_argument('--cost-weight', type=float, default=0.0, help="IK求解位置成本权重，默认：0.0")
    parser.add_argument('--buffer-size', type=int, default=BUFFER_SIZE, help=f"TCP接收缓冲区大小，默认：{BUFFER_SIZE}")
    # 解析参数并返回
    return parser.parse_args()

########################### 重写主函数：适配新通信协议 #########################################
def main():
    # 第一步：解析命令行参数
    args = parse_args()
    # 初始化ROS节点
    rospy.init_node('chess_piece_manipulator', anonymous=True)
    rospy.loginfo("===== 机械臂服务端启动，等待从机连接 =====")
    rospy.loginfo(f"加载配置：监听地址={args.host}, 端口={args.port}, 工作臂={args.arm_side}, 权重={args.cost_weight}")

    # 初始化TCP服务端（使用命令行参数）
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(5)
    rospy.loginfo(f"监听端口 {args.port}，等待从机数据...")

    # 退出标记，收到end指令后触发
    shutdown_flag = False

    # 初始化抓取计数器
    right_num = 1  # 右侧从1开始
    left_num = 1   # 左侧从1开始

    control_leju_claw = init_robot(args.arm_side)

    try:
        while not rospy.is_shutdown() and not shutdown_flag:
            # 阻塞等待连接
            conn, addr = server_socket.accept()
            rospy.loginfo(f"已连接从机：{addr}")

            # 接收数据
            data = conn.recv(args.buffer_size)
            if not data:
                conn.close()
                continue

            # ===================== 核心修改：解析新格式数据包 =====================
            try:
                # 反序列化带指令类型的数据包
                recv_data = pickle.loads(data)
                cmd_type = recv_data[0]

                # 处理【结束指令】：棋局结束，主机安全退出
                if cmd_type == "end":
                    rospy.loginfo("===== 收到从机结束指令，准备退出程序 =====")
                    # 回复从机确认信号
                    conn.send(pickle.dumps("server_shutting_down"))
                    conn.close()
                    # 触发退出标记，跳出循环执行复位
                    shutdown_flag = True
                    continue

                # 处理【落子指令】：正常执行机械臂任务
                elif cmd_type == "move":
                    # 忽略传入的 piece_id，使用自动分配的
                    _, board_row, board_col = recv_data[1], recv_data[2], recv_data[3]
                    rospy.loginfo(f"接收到落子指令：坐标=({board_row},{board_col})")

                    # 参数合法性校验
                    valid_flag = True
                    if not (1 <= board_row <= 3 and 1 <= board_col <= 3):
                        rospy.logwarn(f"拒绝执行：棋盘坐标越界 ({board_row},{board_col})")
                        valid_flag = False
                    elif (board_row, board_col) not in BOARD_POSITIONS:
                        rospy.logwarn(f"拒绝执行：棋盘坐标未配置 ({board_row},{board_col})")
                        valid_flag = False

                    # 执行任务
                    if valid_flag:
                        target_arm_side = select_arm_side(board_row, board_col, args.arm_side)
                        
                        # 根据机械臂侧选择并更新计数器
                        if target_arm_side == 'left':
                            slot_num = left_num
                            left_num = left_num % 3 + 1  # 循环1-3
                        else:
                            slot_num = right_num
                            right_num = right_num % 3 + 1  # 循环1-3
                        
                        # 解析棋子位置
                        try:
                            slot, piece_pos = resolve_piece_position(target_arm_side, slot_num)
                            rospy.loginfo(f"自动分配棋子：{target_arm_side}侧 slot={slot}，位置={piece_pos}")
                            
                            # 执行抓取和放置
                            if not grasp_piece(control_leju_claw, slot_num, target_arm_side):
                                rospy.logerr("抓取失败")
                                conn.send(pickle.dumps("grasp_failed"))
                            elif not place_piece(control_leju_claw, (board_row, board_col), target_arm_side):
                                rospy.logerr("放置失败")
                                conn.send(pickle.dumps("place_failed"))
                            else:
                                # 手臂复位
                                publish_arm_target_poses([1.5], _compose_two_arm_deg(READY_LEFT_ARM_JOINT_DEG, READY_RIGHT_ARM_JOINT_DEG))
                                grip_all(hand=target_arm_side, grip_value=hand_open_value, delay=1.0)
                                rospy.loginfo("任务执行完成！")
                                conn.send(pickle.dumps("task_done"))
                                
                        except ValueError as e:
                            rospy.logerr(f"棋子位置解析失败：{str(e)}")
                            conn.send(pickle.dumps("position_error"))
                    else:
                        conn.send(pickle.dumps("invalid_command"))
                
                # 未知指令类型
                else:
                    rospy.logwarn(f"收到未知指令类型：{cmd_type}")
                    conn.send(pickle.dumps("unknown_command"))

            except pickle.UnpicklingError:
                rospy.logerr("数据反序列化失败，指令格式错误")
                conn.send(pickle.dumps("format_error"))
            except Exception as e:
                rospy.logerr(f"处理指令异常：{str(e)}")
                conn.send(pickle.dumps("server_error"))
            # ====================================================================

            conn.close()
            rospy.loginfo("单次指令处理完成，等待下一个指令...")

    except KeyboardInterrupt:
        rospy.loginfo("服务端被手动关闭")
    finally:
        # ===================== 统一执行复位流程 =====================
        rospy.loginfo("开始恢复机器人初始状态...")

        # 直接回到初始安全姿态（不再按原轨迹倒放）
        publish_arm_target_poses(trajectory_times, trajectory_values)
        time.sleep(1.6)
        
        # 头部复位
        set_head_target(0, 0)
        time.sleep(1)
        
        # 灵巧手张开
        grip_all(hand='both', grip_value=hand_open_value, delay=1.0)
        
        # 恢复手臂控制模式
        set_arm_control_mode(1)
        
        # 关闭套接字，退出ROS
        server_socket.close()
        rospy.loginfo("===== 机器人已恢复初始状态，程序结束 =====")
        rospy.signal_shutdown("服务端正常退出")

if __name__ == '__main__':
    main()