# SAC-RCBF 

Repository containing the code for the paper ["Safe Reinforcement Learning using Robust Control Barrier Functions"](https://arxiv.org/abs/2110.05415). Specifically, an implementation of SAC + Robust Control Barrier Functions (RCBFs) for safe reinforcement learning in multiple custom environments.

While exploring, an RL agent can take actions that lead the system to unsafe states. Here, we use a differentiable RCBF safety layer that minimially alters (in the least-squares sense) the actions taken by the RL agent to ensure the safety of the agent.

<!-- ![Framework Overview](https://github.com/yemam3/SAC-RCBF/raw/master/figures/framework_diagram.png width="100" height="100" "Framework Overview") -->

## Robust Control Barrier Functions (RCBFs)

In this work, we focus on RCBFs that are formulated with respect to differential inclusions of the following form:

<p align="center">
<img src="https://github.com/yemam3/SAC-RCBF/raw/master/figures/diff_inc.png" width=30% height=30%>
</p>
$$\dot{x} \in f(x) + g(x)u + D(x)$$

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

There are two sets of experiments in the paper. The first set evaluates the sample efficiency of SAC-RCBF in two custom environments. The second set evaluates the efficacy of the proposed Modular SAC-RCBF approach at learning the reward-driven task independently from the safety constraints, which results in better transfer performance.

### Experiment 1.1 (Sample Efficiency - Unicycle Env)
* Baseline:
`python main.py --cuda --env Unicycle --cbf_mode baseline --max_episodes 200 --seed 12345`
* Baseline w/ comp:
`python main.py --env Unicycle --cuda --cbf_mode baseline --use_comp True --max_episodes 200 --seed 12345`
* MF SAC-RCBF:
`python main.py --cuda --env Unicycle --cbf_mode full --max_episodes 200 --seed 12345`
* MB SAC-RCBF:
`python main.py --cuda --env Unicycle --model_based --updates_per_step 2 --batch_size 512 --rollout_batch_size 5 --real_ratio 0.3 --gp_max_episodes 70 --cbf_mode full --max_episodes 200 --seed 12345`

### Experiment 1.2 (Sample Efficiency - Simulated Cars Env)
* Baseline:
`python main.py --cuda --env SimulatedCars --max_episodes 300 --cbf_mode baseline --seed 12345`
* Baseline w/ comp:
`python main.py --env SimulatedCars --cuda --cbf_mode baseline --use_comp True --max_episodes 300 --seed 12345`
* MF SAC-RCBF:
`python main.py --cuda --env SimulatedCars --max_episodes 300 --cbf_mode full --seed 12345`
* MB SAC-RCBF:
`python main.py --cuda --env SimulatedCars --model_based --updates_per_step 2 --batch_size 512 --rollout_batch_size 5 --real_ratio 0.3 --max_episodes 300 --cbf_mode full --gp_max_episodes 70 --seed 12345`

### Experiment 2.1 (Modular Learning - Unicycle)
* SAC w/o obstacles (upper performance upper bound):
`python main.py --cuda --env Unicycle --cbf_mode off --rand_init True --obs_config none --seed 12345`
* Modular SAC-RCBF:
`python main.py --cuda --env Unicycle --cbf_mode mod --rand_init True --seed 12345`
* SAC-RCBF:
`python main.py --cuda --env Unicycle --cbf_mode full --rand_init True --seed 12345`
* Baseline:
`python main.py --cuda --env Unicycle --cbf_mode baseline --rand_init True --seed 12345`

* Test zero-shot transfer:
`python main.py --mode test --validate_episodes 200 --resume [run #] --cbf_mode baseline --env Unicycle --obs_config random --seed 12345`

### Experiment 2.2 (Modular Learning - Pvtol)
* SAC w/o obstacles/safety operator (upper performance upper bound):
`python main.py --cuda --env Pvtol --rand_init True --cbf_mode baseline --rand_init True --obs_config none --seed 12345`
* Modular SAC-RCBF:
`python main.py --cuda --env Pvtol --rand_init True --cbf_mode mod --seed 12345`
* SAC-RCBF:
`python main.py --cuda --env Pvtol --rand_init True --cbf_mode full --seed 12345`
* Baseline:
`python main.py --cuda --env Pvtol --rand_init True --cbf_mode baseline --seed 12345`

* Test zero-shot transfer: 
`python main.py --mode test --validate_episodes 200 --resume [run #] --cbf_mode baseline --env Pvtol --obs_config random --seed 12345`
