import serial
import time
import onnxruntime as ort
import numpy as np
import json
from enum import Enum
import psutil
import os

import gymnasium as gym
from gymnasium import spaces

latest_obs = None
data = None
flag_start_obs = 0
Truncated = 0
Failed = 0
Reference = 0

import torch

message_type = 0

if(message_type == 1):
    FIXED_MESSAGE_LENGTH = 18  # Adjust this to your actual fixed message length
else:
    FIXED_MESSAGE_LENGTH = 20 + 14

last_state = None
last_action = None
last_action_control = None
last_ref = None

# Open serial port
ser = serial.Serial(
    port='COM3',
    baudrate=230400,
    timeout=0.00001,                # non-blocking read
    write_timeout=0.00001,        # don't block on write
    inter_byte_timeout=0.00001,   # read returns if no new byte within 10ms
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    rtscts=0,                # Disable RTS/CTS flow control
    xonxoff=0                # Disable software flow control   
)

flag_debug = False # Flag to control print_debuging of debug messages

def print_debug(arg):
    global flag_debug
    if flag_debug:
        print(arg)

last_message_rcv_time = 0 # Variable to store the last message received time
Flag_Stop = 0 #Flag for error detected
Flag_message_received = 0
start_process = False

START_BYTE = 0xD4
END_BYTE = 0xB1

MAX_FORCE = 0.8  # Maximum force value

MAX_FORCE_DERIVATIVE = 1*0.8  # Maximum derivative of force value
MAX_INPUT_DERIVATIVE = 1*0.8  # Maximum derivative of input value

MAX_EPISODE_TIME = 18000  # Maximum episode time in seconds
MAX_POS_TIMEOUT = 8000

MAX_COMM_TIMEOUT = 10  # Maximum communication timeout in seconds

CMD_RECEIVING_DATA_OBS = 0x01
CMD_ABORT_EPISODE = 0x02
CMD_START_EPISODE = 0x03
CMD_STOP_EPISODE = 0x04
CMD_START_TRAINING = 0x05
CMD_STOP_TRAINING = 0x06
CMD_ACKNOWLEDGE = 0x07

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_debug(device)
    return device

class SerialForceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        global start_process
        global normalizing_constants_model

        self.device = get_device()

        self.episode_reward = 0
        self.episode_reward_q = 0
        self.episode_reward_r1 = 0
        self.episode_reward_r2 = 0
        self.episode_reward_r3 = 0

        gain = 1


        self.Q = gain*3
        self.R1 = 0*gain*10e-2
        self.R2 = 0*gain*10e-4
        self.R3 = 0*gain*10e-3

        #self.Q = 1TÇTÇT
        #self.R1 = 5
        #self.R2 = 5
        #self.R3 = 5
        

        self.last_error = 0

        self.keys = ['F_actuator', 'F_actuator_deriv', 'F_load', 'F_load_deriv',
                'Pa', 'Pb', 'x', 'x_deriv']  # last_x shares x scale

        self.last_action = np.zeros(2, dtype=np.float32)  # Initialize last action

        number_action = 2
        number_obs = 12

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(number_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(number_action,), dtype=np.float32)

        self.session = ort.InferenceSession("sac_policy_model.onnx")

        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)

        json_path_model = 'experimentsdata_20k_normalizing_constants.json'
        with open(json_path_model, 'r') as file:
            normalizing_constants_model = json.load(file)

        get_norm_model = lambda key: (
            normalizing_constants_model[key]['min'],
            normalizing_constants_model[key]['max']
        )

        time.sleep(2)  # Wait for connection to settle
        print("Inputs:")
        for i in session.get_inputs():
            print(f"  name: {i.name}, shape: {i.shape}, type: {i.type}")

        print("Outputs:")
        for o in session.get_outputs():
            print(f"  name: {o.name}, shape: {o.shape}, type: {o.type}")

        print("Press 't' to start.")
        start_process = False

        # Internal state
        self.last_state = np.zeros(14, dtype=np.float32) + 0.5

    def reset(self):
        # You should implement a reset signal to the hardware
        global latest_obs
        latest_obs = None

        print("Mean Episode Reward:", self.episode_reward/1637)
        print("Mean Episode Reward Q:", self.episode_reward_q/1637)
        print("Mean Episode Reward R1:", self.episode_reward_r1/1637)
        print("Mean Episode Reward R2:", self.episode_reward_r2/1637)
        print("Mean Episode Reward R3:", self.episode_reward_r3/1637)

        self.episode_reward = 0
        self.episode_reward_q = 0
        self.episode_reward_r1 = 0
        self.episode_reward_r2 = 0
        self.episode_reward_r3 = 0
        #self.send_start_episode()
        obs = self.wait_for_obs()
        self.last_state = obs
        self.last_time = 0
        info  = self._get_info()
        return obs, info
    
    def _get_info(self):
        return {"info": "Placeholder"}

    def step(self, action):
        global Truncated, Failed, Reference
        global data, latest_obs
        global normalizing_constants_model
        # Send action to hardware
        #action = (0.5)*action + self.last_action*0.5  # Smooth action. 

        self.send_action(action)
        #print(time.time() - self.last_time)
        # Wait for new data
        obs = self.wait_for_obs()

        # Compute reward (example: negative tracking error)
        reward_boost = 0

        min_force, max_force = normalizing_constants_model['F_load']['min'], normalizing_constants_model['F_load']['max']

        desired_force = Reference * (max_force - min_force) + min_force 
        F_load = obs[2] * (max_force - min_force) + min_force  # Scale back to original range
        force_error = desired_force - F_load
        force_error_norm = Reference - obs[2]  # Normalized error


        print_debug(f"Desired Force: {desired_force}, F_load: {F_load}, Force Error: {force_error}")
        print_debug(f"Desired Force Norm: {Reference}, F_load Norm: {obs[2]}, Force Error : {force_error_norm}")

        if(abs(desired_force - F_load) < 5):
            reward_boost = 10 - abs(desired_force - F_load)
            reward_boost = 0*1

        max_kp = 10
        max_ki = 10

        reward = -(
            (force_error ** 2) * self.Q
        )

        print_debug(f"Reward: {reward}")
        print_debug(f"Action 0: {(action[0]/2 + 0.5)*(10)}, Action 1: {(action[0]/2 + 0.5)*(6)}")

        reward += 0

        self.episode_reward += reward
        self.episode_reward_q += -(force_error ** 2) * self.Q
        self.episode_reward_r1 += -((action[0]/2 + 0.5)*max_kp ** 2) * self.R1 - ((action[1]/2 + 0.5)*max_ki ** 2) * self.R1
        self.episode_reward_r2 += -((force_error_norm - self.last_error) ** 2) * self.R2
        self.episode_reward_r3 += -(((action[0]/2 + 0.5)*max_kp - (self.last_action[0]/2 + 0.5)*max_kp) ** 2) * self.R3  -(((action[1]/2 + 0.5)*max_ki - (self.last_action[1]/2 + 0.5)*max_ki) ** 2) * self.R3

        self.episode_reward = np.nan_to_num(self.episode_reward, nan=0.0, posinf=0.0, neginf=0.0)
        self.episode_reward_q = np.nan_to_num(self.episode_reward_q, nan=0.0, posinf=0.0, neginf=0.0)
        self.episode_reward_r1 = np.nan_to_num(self.episode_reward_r1, nan=0.0, posinf=0.0, neginf=0.0)
        self.episode_reward_r2 = np.nan_to_num(self.episode_reward_r2, nan=0.0, posinf=0.0, neginf=0.0)
        self.episode_reward_r3 = np.nan_to_num(self.episode_reward_r3, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_action = action

        self.last_error = force_error_norm
        self.last_state = obs
        info = self._get_info()
        #self.last_time = time.time()
        return obs, reward, Failed, Truncated, info

    def send_action(self, action):
        if np.isnan(action).any():
            action = np.zeros(2, dtype=np.float32)

        # Convert [-1, 1] to [0, 255] for two channels
        action_scaled = [int(np.clip(((action[0] + 1) / 2) * 255, 0, 255)),
                         int(np.clip(((action[1] + 1) / 2) * 255, 0, 255))]
        SendMessage(CMD_ACKNOWLEDGE, 2, action_scaled)

    def wait_for_obs(self):
        global start_process
        global latest_obs, flag_start_obs
        global last_message_rcv_time
        global data

        # Blocks until new observation received and parsed

        while True:
            # Process all bytes currently in the queue
            while not serial_queue.empty():
                #start = time.perf_counter()
                full_msg = serial_queue.get()
                process_full_message(full_msg)

                flag_start = received_struct.data[26]

                if latest_obs is not None and flag_start == 1:
                    return latest_obs.copy()
                else:
                    SendMessage(CMD_ACKNOWLEDGE, 2, [0, 0])
            

            # Handle keyboard input (non-blocking, poll every iteration)
            if not start_process and keyboard.is_pressed('t'):
                start_process = True
                last_message_rcv_time = time.time()
                print("Starting...")
                SendMessage(CMD_START_TRAINING, 2, [0,0])
                SendMessage(CMD_START_TRAINING, 2, [0,0])

            if start_process and keyboard.is_pressed('ç'):
                last_message_rcv_time = 0

            # Check for comm timeout
            if start_process and (time.time() - last_message_rcv_time) > MAX_COMM_TIMEOUT:
                print("No response from device. Stopping.")
                SendMessage(CMD_STOP_TRAINING, 2, [0,0])
                start_process = False

            if Flag_Stop == 1:
                print("Stopping program due to Comm Error.")
                break


            time.sleep(0.001)  # N REMOVER


# Get current process


def check_data_for_error(data):
    return_value = 0

    data = 2*data - 1

    if(abs(data[0]) > MAX_FORCE):
        return_value = 1
        print_debug("Force value out of range. Aborting.")
        return return_value
    if(abs(data[1]) > MAX_FORCE_DERIVATIVE):
        return_value = 1
        print_debug("Force derivative out of range. Aborting.")
        return return_value

    return return_value

def CheckEpisodeTimeout(data):
    return_value = 0

    if(data[3] > MAX_EPISODE_TIME):
        return_value = 1
        print_debug("Episode time out of range. Stoping Episode.")
        return return_value

    return return_value


class ReceiveStep(Enum):
    StartByteStep = 1
    LengthStep = 2
    CmdStep = 3
    DataStep = 4
    CrcStep = 5
    EndByteStep = 6

class ReceivedStruct():
    def __init__(self):
        self.start_byte = 0
        self.length = 0
        self.cmd = 0
        self.data = []
        self.crc = 0
        self.end_byte = 0
        self.state = ReceiveStep.StartByteStep


received_struct = ReceivedStruct()

import struct

def bytes_to_uint16(byte_list):
    if len(byte_list) % 2 != 0:
        raise ValueError("Byte list length must be even to form uint16 values.")
    
    result = []
    for i in range(0, len(byte_list), 2):
        low = byte_list[i]
        high = byte_list[i + 1]
        value = (high) | (low << 8)  # Big-endian
        result.append(value)
    return result

Global_CRC = 0

def bytes_to_uint32(data):
    return int.from_bytes(data, byteorder='big', signed=False)

def uint32_to_bytes(value):
    return value.to_bytes(4, byteorder='big', signed=False)

def calculate_crc(data):
    global Global_CRC
    
    try:
        for byte in data:
            Global_CRC = int((Global_CRC + byte)/2) & 0xFF
    except:
        Global_CRC = int((Global_CRC + data)/2) & 0xFF
        
    return Global_CRC

test_input = torch.zeros(1, 14)  # shape: [1, obs_dim]

def ProcessMessage(cmd):
    global last_state, last_action, last_action_control, last_ref, message_type, flag_start_obs, latest_obs, Reference
    global Flag_Stop
    global Flag_message_received
    global Truncated, Failed

    if(True):
            data = received_struct.data

            def extract_uint16(byte_slice):
                values = bytes_to_uint16(byte_slice)
                return values

            def extract_normalized_uint16(byte_slice):
                values = bytes_to_uint16(byte_slice)
                return (np.array(values, dtype=np.uint16) / 65535.0).astype(np.float32)

            # Extract fields (2 bytes per value)
            state = extract_normalized_uint16(data[0:16])           # 8 values (16 bytes)
            ref = extract_normalized_uint16(data[16:18])            # 1 value (2 bytes)
            action_control = extract_normalized_uint16(data[18:20])         # 1 value (2 bytes)
            action = extract_normalized_uint16(data[20:22]) # 1 value (2 bytes)
            action_rl = extract_normalized_uint16(data[22:24])      # 1 value (2 bytes)
            last_ref = extract_normalized_uint16(data[24:26]) 
            flag_start = data[26]
            experiment_time = extract_uint16(data[27:29])[0] 

 

            Reference = ref

            #latest_obs = state.copy()
            flag_start_obs = flag_start

            Failed = 0
            Truncated = 0

            print_debug(flag_start)
            print_debug(experiment_time)

            if(flag_start == 1):    
                if(experiment_time > MAX_EPISODE_TIME):
                    print_debug("Episode time out of range. Stoping Episode.")
                    SendMessage(CMD_STOP_EPISODE, 2, [0,0])
                    Truncated = 1

                if(check_data_for_error(state) == 1):
                    print_debug("Data error detected. Aborting episode.")
                    #SendMessage(CMD_STOP_EPISODE, 2, [0,0])
                    Failed = 1


            if(flag_start == 0):
                if(experiment_time > (MAX_POS_TIMEOUT)):
                    print_debug("Positioning time out of range. Starting Episode.")
                    SendMessage(CMD_START_EPISODE, 2, [0,0])



            # Initialize last_* if this is first run
            if last_state is None:
                last_state = state.copy()
                last_action = action.copy()
                last_action_control = action_control.copy()

            # Prepare model input features
            input_features = np.concatenate([
                state,                                 # shape (8,)
                ref,                                   # shape (1,)
                ref - last_ref,                        # shape (1,)
                ref - state[2:3],                      # shape (1,) for consistency
                action,                                # shape (1,)
            ])

            input_features = (input_features)

            latest_obs = input_features.copy()

            last_state = state
            last_action = action
            last_action_control = action_control

    else:
        print_debug("Unknown command received")

def SendMessage(cmd, length ,data):
    global Flag_Stop
    global Flag_message_received
    global Global_CRC

    message =  [START_BYTE, length, cmd] + data + [END_BYTE] 
    # Initialize message with zeros. Send junk to sync the state machines

    crc_dummy = Global_CRC
    Global_CRC = 0
    crc = calculate_crc(message[0:-1])
    Global_CRC = crc_dummy  # Restore the global CRC value

    message.insert(5, crc + 1)  # Insert CRC after START_BYTE, length, cmd and data

    #message = [0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09, 0x0A, 0x0B, 0x0C,0x0D,0x0E, 0x0F, 0x10]  # Add junk to sync the state machines
    ser.write(bytearray(message))
    ser.flush()  # Ensure all data is sent immediately

    #Wait for response to make sure the message was sent correctly

    #time.sleep(0.1)
    #message_sent = 0
    #start_time = time.time()
    #Flag_message_received = 0

    #while(ser.in_waiting > 0 or message_sent == 0):
    #    incoming_data = ser.read(ser.in_waiting)
    #    print_debug("Received:", incoming_data)   
    #    ReceiveStateMachine(incoming_data[0])
    #    if(Flag_message_received == 1):
    #        message_sent = 1
    #        print_debug("Message sent successfully")
    #    if(time.time() - start_time > 0.1):
    #        ser.write(bytearray(message))
    #        print_debug("Message sending timed out")
            
    #    if(time.time() - start_time > 0.2):
    #        print_debug("Message sending failed")
    #        Flag_Stop = 1
    #        break

    print_debug("Message sent:")
    print_debug(message)

def ReceiveStateMachine(data):
    global Global_CRC
    global Flag_message_received
    global last_message_rcv_time

    match received_struct.state:
        case ReceiveStep.StartByteStep:
            if data == START_BYTE:
                received_struct.data = []  # Reset data buffer
                received_struct.crc = 0
                Global_CRC = 0
                received_struct.start_byte = data
                calculate_crc(received_struct.start_byte)
                received_struct.state = ReceiveStep.LengthStep
                print_debug("Start byte received")
            else:
                print_debug("Invalid start byte")

        case ReceiveStep.LengthStep:
            received_struct.length = data
            calculate_crc(received_struct.data)
            received_struct.state = ReceiveStep.CmdStep
            print_debug("Length byte received")

        case ReceiveStep.CmdStep:
            received_struct.cmd = data
            calculate_crc(received_struct.data)
            if(received_struct.length > 0):
                received_struct.state = ReceiveStep.DataStep
            else:
                received_struct.state = ReceiveStep.CrcStep
            print_debug("Command byte received")

        case ReceiveStep.DataStep:
            if len(received_struct.data) < received_struct.length:
                received_struct.data.append(data)
                calculate_crc(received_struct.data)
                if len(received_struct.data) == received_struct.length:
                    received_struct.state = ReceiveStep.CrcStep
                    print_debug("Data bytes received")
            else:
                print_debug("Data buffer overflow")

        case ReceiveStep.CrcStep:
            received_struct.crc = data

            Global_CRC = Global_CRC + 1

            if(Global_CRC == received_struct.crc):
                print("CRC check passed")
                received_struct.state = ReceiveStep.EndByteStep
            else:
                print("CRC check failed")
                print(Global_CRC)
                print(received_struct.crc)
                received_struct.state = ReceiveStep.EndByteStep
                received_struct.crc = 0
                

        case ReceiveStep.EndByteStep:
            if data == END_BYTE:
                received_struct.end_byte = data
                print_debug("End byte received")
                # Process the complete message here
                Flag_message_received = 1
                last_message_rcv_time = time.time()
                ProcessMessage(received_struct.cmd)
                Flag_message_received = 0
                received_struct.state = ReceiveStep.StartByteStep
                received_struct.crc = 0
                received_struct.data = []
            else:
                received_struct.state = ReceiveStep.StartByteStep
                print_debug("Invalid end byte")


import keyboard
import threading
import queue
import time
import keyboard

# Create a queue to hold incoming bytes from serial port
serial_queue = queue.Queue()

def serial_read_thread():
    buffer = bytearray()
    while True:
        try:
            if ser.in_waiting > 0:
                buffer += ser.read(ser.in_waiting)

                while len(buffer) >= FIXED_MESSAGE_LENGTH:
                    # print(buffer)
                    # Search for valid start and end byte positions
                    if buffer[0] == START_BYTE and buffer[FIXED_MESSAGE_LENGTH - 1] == END_BYTE:
                        full_msg = buffer[:FIXED_MESSAGE_LENGTH]
                        buffer = buffer[FIXED_MESSAGE_LENGTH:]
                        serial_queue.put(full_msg)
                    else:
                        # Drop first byte (misaligned or garbage)
                        buffer.pop(0)


        except serial.SerialException:
            print("Serial connection error in thread")
            break
        except Exception as e:
            print("Unexpected error in serial thread:", e)
            break


def improved_calculate_crc(byte):
    global Global_CRC
    Global_CRC = (Global_CRC + byte) & 0xFF
    return Global_CRC

# Replace your calculate_crc with improved version, and fix usage accordingly
#calculate_crc = improved_calculate_crc

# Start serial thread before main loop
serial_thread = threading.Thread(target=serial_read_thread, daemon=True)
serial_thread.start()


def process_full_message(msg):
    global last_message_rcv_time, Flag_message_received, Global_CRC

    start, length, cmd = msg[0], msg[1], msg[2]
    data = list(msg[3:-2])
    crc = msg[-2]
    end = msg[-1]

    # Basic CRC check
    dummy_crc = Global_CRC
    Global_CRC = 0
    computed_crc = calculate_crc(data)
    Global_CRC = dummy_crc

    if computed_crc != (crc - 1):
        print_debug("CRC check failed")
        #return

    received_struct.cmd = cmd
    received_struct.data = data
    Flag_message_received = 1
    last_message_rcv_time = time.time()

    ProcessMessage(cmd)
    Flag_message_received = 0

import onnx

model = onnx.load("sac_policy_model.onnx")

session = ort.InferenceSession("sac_policy_model.onnx")
input_data = np.random.randn(1, 14).astype(np.float32)

from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.sac import SAC
from skrl.memories.torch import RandomMemory



import gymnasium as gym
from gymnasium.envs.registration import register
import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed
    
# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-2, max_log_std=3, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.dense1_bn = nn.Identity(32)
        self.dense2_bn = nn.Identity(32)
        self.action_layer = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * 1.5)

    def compute(self, inputs, role):
        x = F.relu((self.linear_layer_1(inputs["states"])))
        x = F.relu((self.linear_layer_2(x)))
        # Pendulum-v1 action_space is -2 to 2
        return 1* torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.linear_layer_3 = nn.Linear(32, 1)
        self.dense1_bn = nn.Identity(32)
        self.dense2_bn = nn.Identity(32)


    def compute(self, inputs, role):
        x = F.relu((self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1))))
        x = F.relu((self.linear_layer_2(x)))
        return self.linear_layer_3(x), {}
        


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version

if __name__ == '__main__':
    register(
        id="ForceControl_BB_Lyap-v0",
        entry_point="src.envs:ForceControlBenchEnvironment_BB_Lyap"
    )

    register(
        id="ForceControl_BB_Lyap_RL_control-v0",
        entry_point="src.envs:ForceControlBenchEnvironment_BB_Lyap_FL_Rl_control"
    )
    
    env = SerialForceEnv()
    env = wrap_env(env)

    device = env.device


    # instantiate a memory as experience replay
    memory = RandomMemory(memory_size=1000000, num_envs=env.num_envs, device=device, replacement=True)


    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models

    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["discount_factor"] = 0.98
    #cfg["batch_size"] = 64
    cfg["batch_size"] = 32
    cfg["random_timesteps"] = 1   # collect random actions first
    cfg["learning_starts"] = 1
    #cfg["learn_entropy"] = True
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 500
    cfg["experiment"]["checkpoint_interval"] = 500

    #cfg["initial_entropy_value"] = 0.05   #   ←  low temperature
    #cfg["learn_entropy"]        = False   #   ←  freeze it (no tuning)


    #Valores Novos
    cfg["initial_entropy_value"] = 1
    cfg["learn_entropy"] = True       # ← freeze it at 0.2

    cfg["discount_factor"] = 1
    cfg["target_entropy"] = -1.0   # very high exploration
    cfg["learn_entropy"] = True 

    cfg["experiment"]["write_interval"] = 1637
    cfg["experiment"]["checkpoint_interval"] = 1637
    cfg["experiment"]["directory"] = "runs/torch/HyD"
    cfg["experiment"]["name"] = "SAC_ForceControlBench"
    cfg["experiment"]["write_tensorboard"] = True

    cfg["actor_learning_rate"] = 0.001 #default:0.001
    cfg["critic_learning_rate"] = 0.001 #default:0.001
    cfg["entropy_learning_rate"] = 0.001 #default:0.001

    #models["policy"].load_state_dict(torch.load("runs/torch/HyD/SAC_ForceControlBench/policy.pth", map_location=device))
    #models["critic_1"].load_state_dict(torch.load("runs/torch/HyD/SAC_ForceControlBench/critic_1.pth", map_location=device))
    #models["critic_2"].load_state_dict(torch.load("runs/torch/HyD


    agent = SAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
    
    path_checkpint = ""

    agent.load(path_checkpint + "slide_pretrain_1.pt")
    #agent.load(path_checkpint + "runs/torch/HyD/slide_online_2/checkpoints/best_agent.pt")

    # configure and instantiate the RL trainert
    cfg_trainer = {"timesteps": 4*100000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()
