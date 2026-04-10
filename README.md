# Bio-Inspired Robot Navigation using Spiking Neural Networks

This project implements a sensory-motor control system for robot navigation using a **Spiking Neural Network (SNN)**. The system simulates a differential-drive robot equipped with multiple directional sensors operating in a 2D environment with obstacles.

The approach is based on **temporal spike-based computation**, inspired by biological neural systems.

---

## System Components

### 1. Sensor Model

The robot has N = 7 directional sensors distributed between -pi/2 and +pi/2.

Each sensor outputs a normalized distance:
s_i = d_i / d_max
where:
* d_i = distance to nearest obstacle
* d_max = maximum sensor range
* s_i ∈ [0, 1]

---

### 2. Spike Encoding

Sensor values are converted into spike trains using stochastic rate encoding:
x_i(t) ~ Bernoulli(s_i)

This produces a binary spike sequence:
X ∈ {0,1}^{T × N}

where:
* T = temporal sequence length
* N = number of sensors

---

### 3. Spiking Neural Network (SNN)

The network uses Leaky Integrate-and-Fire (LIF) neurons.

#### Membrane Potential Update

V(t+1) = beta * V(t) + I(t)
where:
* V(t) = membrane potential
* beta ∈ (0,1) = decay factor
* I(t) = input current

#### Spike Generation

S(t) = 1 if V(t) ≥ V_th, else 0
After a spike:
V(t) = V(t) - V_th

---

### 4. Network Architecture

Input (7) → Linear (96) → LIF → Linear (2) → LIF

The network processes a temporal sequence:
x(1), x(2), ..., x(T)

Final output is the membrane state at time T:
y = V_out(T)

---

### 5. Motor Output

The model predicts wheel velocities:
(v_l, v_r)
These are rescaled using a trained scaler.

---

### 6. Robot Kinematics

#### Linear Velocity

v = (v_l + v_r) / 2

#### Angular Velocity

omega = (v_r - v_l) / L
where L is the wheel base.

#### State Update

theta(t+1) = theta(t) + omega * dt
x(t+1) = x(t) + v * cos(theta) * dt
y(t+1) = y(t) + v * sin(theta) * dt

---

## Simulation Environment

* UI made using Streamlit
* 2D square arena
* Random objects of different shapes and sizes
* Robot learns obstacle avoidance behavior
* Can control the shape, size, number of obstacles as well as the speed, steps etc. of the robot.

---

## Dataset

Generated using simulation with an expert controller.

### Structure

* X: sensor data with shape (E, T, N)
* Y: motor outputs with shape (E, T, 2)
where:
* E = number of episodes
* T = steps per episode
* N = number of sensors

#### Noise Injection

s_tilde = clip(s + Normal(0, sigma^2), 0, 1)

