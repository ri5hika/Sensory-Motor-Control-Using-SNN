# Bio-Inspired Robot Navigation using Spiking Neural Networks

## Overview

This project implements a sensory-motor control system for robot navigation using a **Spiking Neural Network (SNN)**. The system simulates a differential-drive robot equipped with multiple directional sensors operating in a 2D environment with obstacles.

The core idea is to model control using **temporal spike-based computation**, inspired by biological neural systems, instead of traditional continuous-valued neural networks.

---

## System Components

### 1. Sensor Model

The robot is equipped with ( N = 7 ) directional range sensors distributed over an angular field:

[
\theta_i \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]
]

Each sensor returns a normalized distance:

[
s_i = \frac{d_i}{d_{\text{max}}}, \quad s_i \in [0,1]
]

where:

* ( d_i ) = distance to nearest obstacle along ray
* ( d_{\text{max}} ) = maximum sensor range

---

### 2. Spike Encoding

Sensor readings are converted into spike trains using **rate-based stochastic encoding**:

[
x_i^{(t)} \sim \text{Bernoulli}(s_i)
]

This produces a binary spike sequence:

[
X \in {0,1}^{T \times N}
]

where:

* ( T ) = temporal window length
* ( N ) = number of sensors

---

### 3. Spiking Neural Network (SNN)

The model uses **Leaky Integrate-and-Fire (LIF)** neurons.

#### Membrane Potential Dynamics

[
V(t+1) = \beta V(t) + I(t)
]

where:

* ( V(t) ) = membrane potential
* ( \beta \in (0,1) ) = decay constant
* ( I(t) ) = input current

#### Spike Generation

[
S(t) =
\begin{cases}
1, & \text{if } V(t) \geq V_{\text{th}} \
0, & \text{otherwise}
\end{cases}
]

After spiking:

[
V(t) \leftarrow V(t) - V_{\text{th}}
]

---

### 4. Network Architecture

[
\text{Input (7)} \rightarrow \text{Linear (96)} \rightarrow \text{LIF} \rightarrow \text{Linear (2)} \rightarrow \text{LIF}
]

Temporal processing is performed over a sequence:

[
{x^{(1)}, x^{(2)}, \dots, x^{(T)}}
]

Final membrane state is used as output:

[
y = V_{\text{out}}(T)
]

---

### 5. Motor Output

The network predicts wheel velocities:

[
(v_l, v_r)
]

These are rescaled using a trained scaler.

---

### 6. Robot Kinematics

The robot follows differential drive equations:

#### Linear Velocity

[
v = \frac{v_l + v_r}{2}
]

#### Angular Velocity

[
\omega = \frac{v_r - v_l}{L}
]

where ( L ) is the wheel base.

#### State Update

[
\theta_{t+1} = \theta_t + \omega \Delta t
]

[
x_{t+1} = x_t + v \cos(\theta) \Delta t
]

[
y_{t+1} = y_t + v \sin(\theta) \Delta t
]

---

## Simulation Environment

* 2D square arena
* Randomly generated circular obstacles
* Collision avoidance emerges from learned behavior

---

## Dataset

The dataset is generated via simulation using an expert controller.

### Structure

* ( X \in \mathbb{R}^{E \times T \times N} ): sensor readings
* ( Y \in \mathbb{R}^{E \times T \times 2} ): motor outputs

where:

* ( E ) = number of episodes
* ( T ) = steps per episode
* ( N ) = number of sensors

Noise is injected into sensor readings:

[
\tilde{s} = \text{clip}(s + \mathcal{N}(0, \sigma^2), 0, 1)
]

---

## Execution

### Install dependencies

```
pip install streamlit torch snntorch matplotlib numpy joblib
```

### Run application

```
streamlit run snn5.py
```

---

## Output

* Robot trajectory visualization
* Sensor ray interactions
* Animated simulation (GIF)

---

## Key Characteristics

* Temporal computation using spike trains
* Event-driven processing instead of continuous activations
* Bio-inspired neuron dynamics
* Learned sensor-to-motor mapping

---

## Limitations

* Uses rate-based spike encoding (not fully event-driven)
* Evaluation metrics are heuristic (not task-optimal)
* No reinforcement learning or online adaptation

---

## Future Work

* Replace rate coding with temporal coding schemes
* Introduce reward-based learning (RL + SNN)
* Add collision penalties and goal-directed tasks
* Extend to multi-agent environments

---

## Summary

This project demonstrates how spiking neural networks can be applied to continuous control problems by combining:

* stochastic spike encoding
* leaky integrate-and-fire neuron models
* temporal sequence processing
* differential drive kinematics

The result is a biologically inspired control system capable of obstacle avoidance in a simulated environment.
