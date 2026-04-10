import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib import animation
from collections import deque
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import joblib
import random

# =============================
# CONSTANTS
# =============================
ARENA_SIZE = 8.0
NUM_SENSORS = 7
SEQ_LEN = 20
DT = 0.1
WHEEL_BASE = 0.5
FPS = 25
RENDER_SKIP = 3

BOUNDARY_MARGIN = 0.25
BOUNDARY_TURN_GAIN = np.pi  # 180°

sensor_angles = np.linspace(-np.pi / 2, np.pi / 2, NUM_SENSORS)

# =============================
# SNN MODEL
# =============================
beta = 0.9
spike_grad = surrogate.fast_sigmoid()

class TemporalSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_SENSORS, 96)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(96, 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spike_seq):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for t in range(spike_seq.shape[0]):
            cur1 = self.fc1(spike_seq[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)
        return mem2

# =============================
# LOAD MODEL
# =============================
device = torch.device("cpu")
model = TemporalSNN().to(device)
model.load_state_dict(torch.load("snn_temporal_model.pth", map_location=device))
model.eval()
scaler_y = joblib.load("snn_seq_scaler_y.pkl")

# =============================
# SENSOR FUNCTION
# =============================
def sense(pos, theta, obstacles, sensor_range):
    readings, hit_points = [], []

    for ang in sensor_angles:
        ray = np.array([np.cos(theta + ang), np.sin(theta + ang)])
        best = sensor_range

        for shape, p in obstacles:
            cx, cy = p[0], p[1]

            if shape == "circle":
                r = p[2]
            elif shape == "rect":
                r = np.hypot(p[2], p[3]) / 2
            else:  # triangle
                r = p[2]

            rel = np.array([cx, cy]) - pos
            proj = rel.dot(ray)

            if 0 < proj < best:
                closest = pos + proj * ray
                if np.linalg.norm(np.array([cx, cy]) - closest) < r:
                    best = proj

        readings.append(best / sensor_range)
        hit_points.append(pos + ray * best)

    return np.array(readings), hit_points

# =============================
# HIGH-LEVEL BEHAVIOR
# =============================
def attraction_steering(pos, theta, obstacles, gain=0.15):
    centers = np.array([p[:2] for _, p in obstacles])
    dists = np.linalg.norm(centers - pos, axis=1)
    target = centers[np.argmin(dists)]

    desired = np.arctan2(target[1] - pos[1], target[0] - pos[0])
    err = np.arctan2(np.sin(desired - theta), np.cos(desired - theta))
    return gain * err

def boundary_avoidance(pos):
    if (
        pos[0] < BOUNDARY_MARGIN or
        pos[0] > ARENA_SIZE - BOUNDARY_MARGIN or
        pos[1] < BOUNDARY_MARGIN or
        pos[1] > ARENA_SIZE - BOUNDARY_MARGIN
    ):
        return BOUNDARY_TURN_GAIN
    return 0.0

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(layout="wide")
st.title("Bio-Inspired Sensory-Motor Control Using Spiking Neural Networks")

with st.sidebar:
    st.header("Simulation Controls")

    num_obstacles = st.slider("Number of Obstacles", 1, 20, 8)
    obstacle_size = st.slider("Obstacle Size (Global)", 0.2, 1.2, 0.6)

    shapes = st.multiselect(
        "Obstacle Shapes",
        ["circle", "rectangle", "triangle"],
        default=["circle", "rectangle", "triangle"]
    )

    robot_speed = st.slider("Robot Speed", 0.3, 3.0, 1.3)
    sensor_range = st.slider("Sensor Range", 1.0, 4.0, 2.5)
    steps = st.slider("Simulation Steps", 150, 800, 400)
    seed = st.number_input("Random Seed", value=42)

if not shapes:
    st.error("Select at least one obstacle shape.")
    st.stop()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =============================
# GENERATE OBSTACLES
# =============================
obstacles = []

for _ in range(num_obstacles):
    x = random.uniform(1, ARENA_SIZE - 1)
    y = random.uniform(1, ARENA_SIZE - 1)
    shape = random.choice(shapes)

    if shape == "circle":
        obstacles.append(("circle", (x, y, obstacle_size)))
    elif shape == "rectangle":
        obstacles.append(("rect", (x, y, obstacle_size, obstacle_size)))
    elif shape == "triangle":
        obstacles.append(("tri", (x, y, obstacle_size)))

# =============================
# SIMULATION
# =============================
pos = np.array([ARENA_SIZE / 2, ARENA_SIZE / 2])
theta = random.uniform(-np.pi, np.pi)

hist = deque([np.ones(NUM_SENSORS)], maxlen=SEQ_LEN)
trajectory, sensor_hits = [], []

for _ in range(steps):
    d, hits = sense(pos, theta, obstacles, sensor_range)
    hist.append(d)

    seq = torch.tensor(np.array(hist), dtype=torch.float32).unsqueeze(1)
    spikes = (torch.rand_like(seq) < seq).float()

    with torch.no_grad():
        out = model(spikes)

    v_l, v_r = scaler_y.inverse_transform(out.numpy())[0]
    v_l *= robot_speed
    v_r *= robot_speed

    v = (v_l + v_r) / 2
    omega = (v_r - v_l) / WHEEL_BASE

    steer_bias = attraction_steering(pos, theta, obstacles)
    wall_turn = boundary_avoidance(pos)

    theta += (omega + steer_bias + wall_turn) * DT
    pos += np.array([np.cos(theta), np.sin(theta)]) * v * DT
    pos = np.clip(pos, 0.2, ARENA_SIZE - 0.2)

    trajectory.append(pos.copy())
    sensor_hits.append(hits)

trajectory = np.array(trajectory)[::RENDER_SKIP]
sensor_hits = sensor_hits[::RENDER_SKIP]

# =============================
# ANIMATION
# =============================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, ARENA_SIZE)
ax.set_ylim(0, ARENA_SIZE)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

for shape, p in obstacles:
    if shape == "circle":
        ax.add_patch(Circle((p[0], p[1]), p[2], color="black"))
    elif shape == "rect":
        ax.add_patch(Rectangle((p[0], p[1]), p[2], p[3], color="black"))
    elif shape == "tri":
        ax.add_patch(
            RegularPolygon(
                xy=(p[0], p[1]),
                numVertices=3,
                radius=p[2],
                orientation=np.pi / 2,
                color="black"
            )
        )

traj_line, = ax.plot([], [], "b-", lw=2)
robot_dot, = ax.plot([], [], "ro", ms=7)
sensor_lines = [ax.plot([], [], "r--", lw=1)[0] for _ in range(NUM_SENSORS)]

def animate(i):
    traj_line.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])
    robot_dot.set_data(trajectory[i, 0], trajectory[i, 1])
    for s, h in zip(sensor_lines, sensor_hits[i]):
        s.set_data([trajectory[i, 0], h[0]], [trajectory[i, 1], h[1]])
    return [traj_line, robot_dot] + sensor_lines

ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=40)
ani.save("simulation.gif", writer="pillow", fps=FPS)

st.image("simulation.gif", caption="Robot seeks obstacles, avoids them, and turns at walls")
