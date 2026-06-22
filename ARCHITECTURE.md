# Architecture Overview

This document describes the codebase structure and design of Smart Elevator.

## High-Level Design

```
┌─────────────────────────────────────┐
│   Training Scripts                   │
├─────────────────────────────────────┤
│  single_agent/train.py               │
│  marl/train_marl.py                  │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼─────┐  ┌──────▼─────────────┐
│ Gym Env    │  │ PettingZoo Env     │
│ (Single)   │  │ (Multi-Agent)      │
└──────┬─────┘  └──────┬─────────────┘
       │                │
       └────────┬───────┘
                │
    ┌───────────▼──────────────┐
    │  BaseElevatorEnv          │
    │  (Core Simulation Logic)  │
    └───────────┬───────────────┘
                │
    ┌───────────┴────────────┬──────────┐
    │                        │          │
┌───▼────┐  ┌────────┐  ┌───▼──┐  ┌────▼───┐
│Building │  │Elevator│  │Person│  │  View  │
│ (State) │  │(Entity)│  │(Data)│  │(Render)│
└─────────┘  └────────┘  └──────┘  └────────┘
```

## Core Modules

### 1. **Simulation Engine**

#### `building.py` - Main Simulation State
- Manages all floors, elevators, and waiting people
- Core logic: `step(actions)` - processes elevator commands and updates state
- Tracks metrics: delivered count, wait times, passenger distribution
- Notifies observers of state changes

**Key Classes:**
- `Building` - Orchestrates simulation, manages entities

**Key Methods:**
- `step(actions)` - Executes one simulation step
- `get_elevator_state()` - Returns elevator state info
- `reset()` - Resets simulation to initial state

#### `elevator.py` - Elevator Entity
- Simulates individual elevator physics and behavior
- Tracks: position, direction, passengers, idle time
- Actions: move up/down/stay

**Key Classes:**
- `Elevator` - Represents a physical elevator

**Key Methods:**
- `execute_action(action)` - Processes movement command
- `pick_up_passenger()` - Board waiting passenger
- `drop_off_passenger()` - Release passenger at destination
- `is_idle()` - Check if elevator has no work

#### `person.py` - Person/Passenger Data
- Represents passengers with source/destination floors
- Tracks wait time

**Key Classes:**
- `Person` - Simple data class for passenger info

#### `constants.py` - Configuration
- Screen dimensions, rendering settings

### 2. **Environment Wrappers**

#### `base_env.py` - Shared Logic
- Common environment initialization
- Observation generation (state representation)
- Rendering setup
- Initialization: Building, view, pygame

**Key Classes:**
- `BaseElevatorEnv` - Abstract base for gym/pettingzoo envs

**Key Methods:**
- `_get_obs()` - Build observation from state
- `render()` - Display simulation
- `close()` - Cleanup

#### `environment.py` - Single-Agent Gym Env
- Wraps simulation as OpenAI Gym environment
- One agent controls all elevators

**Key Classes:**
- `SingleAgentElevatorEnv(gym.Env)` - Standard Gym interface

**Key Methods:**
- `reset()` - Initialize simulation
- `step(action)` - Execute action, get obs/reward/done
- Observation: 1D array of elevator + global state

#### `marl/multi_agent_env.py` - Multi-Agent PettingZoo Env
- Wraps simulation as PettingZoo parallel environment
- Each elevator gets its own agent

**Key Classes:**
- `MARLElevatorEnv(ParallelEnv, BaseElevatorEnv)` - PettingZoo interface

**Key Methods:**
- `reset()` - Initialize with multiple agents
- `step(actions)` - Execute concurrent actions
- Returns: obs, rewards, terminations, truncations, infos

### 3. **Visualization**

#### `view.py` - Rendering
- Pygame-based visualization
- Draws: building layout, elevators, people, queues

**Key Classes:**
- `BuildingView` - Handles all rendering

**Key Methods:**
- `draw()` - Render current frame
- `_draw_floor()` - Draw individual floors
- `_draw_elevator()` - Draw elevator sprite

## Observation Space

### Single-Agent Observation
- **Size:** 1 + 3 + num_floors + (2 * num_floors)
- **Components:**
  1. Elevator position (normalized: 0-1)
  2. Direction one-hot (up/idle/down)
  3. Passenger destinations histogram (num_floors bins)
  4. Global waiting people up requests (num_floors)
  5. Global waiting people down requests (num_floors)

### Multi-Agent Observation (Per Agent)
- **Same as single-agent, per elevator**
- Each agent sees its own local state + global queue info

## Action Space

### Single-Agent Actions
- **Type:** Discrete(3)
- **Mapping:** 
  - 0 = Move down
  - 1 = Stay
  - 2 = Move up
- **Note:** In multi-agent, each elevator gets independent action

### Multi-Agent Actions
- **Per agent:** Discrete(3)
- **All agents act concurrently**

## Reward Structure

### Single-Agent
- **Positive:** +50 per passenger dropped off, +10 per pickup
- **Negative:** -1 per idle step with work available, -0.05 per waiting passenger

### Multi-Agent
- **Per-agent reward:** Same as single-agent + system penalty
- **System penalty:** -0.05 * (total_waiting / num_elevators) per agent
- **Purpose:** Encourages collaboration vs. selfish optimization

## Training Infrastructure

### Single-Agent (`single_agent/`)
- **Algorithms:** PPO, DQN (stable-baselines3)
- **Framework:** OpenAI Gym
- **Output:** .zip model files

### Multi-Agent (`marl/`)
- **Algorithm:** PPO with shared policy
- **Framework:** Ray RLlib + PettingZoo
- **Output:** RLlib checkpoint format
- **Architecture:** One shared neural network, multiple agents using same policy

## Configuration

- **`params.json`** - Simulation parameters:
  - num_floors, num_elevators, elevator_capacity
  - Person generation rates and patterns
  - Simulation step size

- **Algorithm configs** - Embedded in train scripts:
  - Learning rates, batch sizes, entropy coefficients
  - Training iterations, checkpointing intervals

## Data Flow During Training

```
1. reset() → Initialize building, get initial obs
2. act() → Agent processes obs, outputs action
3. step(action) → Simulation updates, collects reward
4. Repeat until done
5. save_checkpoint() → Store model weights
```

## Directory Structure

```
Smart-Elevator/
├── building.py              # Simulation state machine
├── elevator.py              # Individual elevator logic
├── person.py               # Passenger data structure
├── base_env.py             # Shared environment logic
├── environment.py          # Single-agent Gym wrapper
├── view.py                 # Pygame visualization
├── constants.py            # Configuration constants
├── params.json             # Simulation hyperparameters
│
├── single_agent/
│   ├── train.py            # PPO/DQN training script
│   ├── single_agent_env.py # Gym environment
│   ├── compare_all_agents.py
│   ├── train_ppo.py
│   ├── train_dqn.py
│   └── utils.py
│
├── marl/
│   ├── train_marl.py       # Multi-agent training
│   ├── evaluate_marl.py    # Multi-agent evaluation
│   ├── multi_agent_env.py  # PettingZoo environment
│   └── README_MARL.md
│
├── models/                 # Trained model checkpoints
├── logs/                   # Training logs (TensorBoard)
│
├── README.md               # Main documentation
├── SETUP.md                # Installation guide
├── CONTRIBUTING.md         # Contribution guidelines
└── .github/workflows/      # CI/CD configuration
```

## Key Design Decisions

1. **Shared Environment Logic** - `BaseElevatorEnv` avoids duplication
2. **Modular Simulation** - `Building`, `Elevator`, `Person` are loosely coupled
3. **Pluggable RL Wrappers** - Easy to add new algorithms or frameworks
4. **Observation Normalization** - Improves training stability
5. **Multi-Agent Penalty** - Discourages selfish behavior in MARL

## Extension Points

**Add a new algorithm:**
1. Create `new_algorithm/` directory
2. Implement environment wrapper (Gym or PettingZoo)
3. Create training script using preferred framework
4. Add to comparison script

**Add new building features:**
1. Extend `Building.step()` with new mechanics
2. Update observation in `BaseElevatorEnv`
3. Adjust rewards if needed

**Change reward structure:**
1. Modify reward calculation in environment wrappers
2. Test convergence with new rewards
3. Document rationale in code comments

## Performance Considerations

- **Rendering overhead:** ~20% slowdown with pygame. Disable for fast training.
- **Multi-agent scalability:** RLlib scales well to 10+ agents, but checkpoint size grows.
- **Memory:** ~2GB for typical 4-floor, 4-elevator setup with training.
- **GPU speedup:** 2-5x faster training with CUDA on modern GPU.
