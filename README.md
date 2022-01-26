# SAC-RCBF 

Repository containing the code for the paper "Safe  Model-Based  Reinforcement  Learning using Robust Control Barrier Functions". Specifically, an implementation of SAC + Robust Control Barrier Functions (RCBFs) for safe reinforcement learning in two custom environments.

While exploring, an RL agent can take actions that lead the system to unsafe states. Here, we use a differentiable RCBF safety layer that minimially alters (in the least-squares sense) the actions taken by the RL agent to ensure the safety of the agent.

<!-- ![Framework Overview](https://github.com/yemam3/SAC-RCBF/raw/master/figures/framework_diagram.png width="100" height="100" "Framework Overview") -->

<p align="center">
<img src="https://github.com/yemam3/SAC-RCBF/raw/master/figures/framework_diagram.png" width=60% height=60%>
</p>

## Robust Control Barrier Functions (RCBFs)

In this work, we focus on RCBFs that are formulated with respect to differential inclusions of the following form:

<p align="center">
<img src="https://github.com/yemam3/SAC-RCBF/raw/master/figures/diff_inc.png" width=30% height=30%>
</p>

Here `D(x)` is a disturbance set unkown apriori to the robot, which we learn online during traing via Gaussian Processes (GPs). The underlying library is GPyTorch. 
 
The QP used to ensure the system's safety is given by:

<p align="center">
<img src="https://github.com/yemam3/SAC-RCBF/raw/master/figures/qp.png" width=70% height=70%>
</p>

where `h(x)` is the RCBF, and `u_RL` is the action outputted by the RL policy. As such, the final (safe) action taken in the environment is given by `u = u_RL + u_RCBF` as shown in the following diagram:

<p align="center">
<img src="https://github.com/yemam3/SAC-RCBF/raw/master/figures/policy_diagram.png" width=70% height=70%>
</p>


## Coupling RL & RCBFs to Improve Training Performance

The above is sufficient to ensure the safety of the system, however, we would also like to improve the performance of the learning by letting the RCBF layer guide the training. This is achieved via:
* Using a differentiable version of the safety layer that allows us to backpropagte through the RCBF based Quadratic Program (QP) resulting in an end-to-end policy.
* Using the GPs and the dynamics prior to generate synthetic data (model-based RL).

## Other Approaches

In addition, the approach is compared against two other frameworks (implemented here) in the experiments:
* A vanilla baseline that uses SAC with RCBFs without generating synthetic data nor backproping through the QP (RL loss computed wrt ouput of RL policy).
* A modified approach from ["End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks"](https://ojs.aaai.org/index.php/AAAI/article/view/4213) that replaces their discrete time CBF formulation with RCBFs, but makes use of the supervised learning component to speed up the learning.

## Running the Experiments

The two environments are `Unicycle` and `SimulatedCars`. `Unicycle` involves a unicycle robot tasked with reaching a desired location while avoiding obstacles and `SimulatedCars` involves a chain of cars driving in a lane, the RL agent controls the 4th car and must try minimzing control effort while avoiding colliding with the other cars.

### `Unicycle` Env: 

* Running the proposed approach: 
`python main.py --env SimulatedCars --gamma_b 20 --max_episodes 400 --cuda --updates_per_step 2 --batch_size 512  --seed 12345 --model_based`

* Running the baseline:
`python main.py --env SimulatedCars --gamma_b 20 --max_episodes 400 --cuda --updates_per_step 1 --batch_size 256  --seed 12345 --no_diff_qp`

* Running the modified approach from "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks": 
`python main.py --env SimulatedCars --gamma_b 20 --max_episodes 400 --cuda --updates_per_step 1 --batch_size 256   --seed 12345 --no_diff_qp --use_comp True`

### `SimulatedCars` Env:

* Running the proposed approach: 
`python main.py --env Unicycle --gamma_b 50 --max_episodes 200 --cuda --updates_per_step 2 --batch_size 512  --seed 12345 --model_based`

* Running the baseline:
`python main.py --env Unicycle --gamma_b 50 --max_episodes 200 --cuda --updates_per_step 1 --batch_size 256  --seed 12345 --no_diff_qp`

* Running the modified approach from "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks": 
`python main.py --env Unicycle --gamma_b 50 --max_episodes 200 --cuda --updates_per_step 1 --batch_size 256   --seed 12345 --no_diff_qp --use_comp True`
