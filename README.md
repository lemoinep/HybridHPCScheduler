# Hybrid HPC Scheduler

HybridHPCScheduler is an experimental framework for the development of an AI-based APIC system.
It is designed to support intelligent scheduling and resource orchestration in heterogeneous HPC environments.


This project implements an advanced hybrid HPC scheduler designed for heterogeneous computing environments. 
It introduces a segment-aware scheduling strategy that captures fine-grained execution characteristics across diverse hardware resources, 
including CPUs, GPUs, and other accelerators.

At its core, the system models a multi-device resource space, enabling efficient allocation and coordination of tasks across heterogeneous architectures. 
The scheduling problem is formulated and solved using a constraint programming approach based on the CP-SAT solver, 
allowing for optimal or near-optimal decisions under complex constraints such as resource contention, task dependencies, and execution segments.

To explore adaptive and learning-based strategies, the project integrates a custom Gymnasium environment that formalizes the scheduling process 
as a reinforcement learning problem. On top of this environment, a Proximal Policy Optimization (PPO) agent is trained to learn 
dynamic scheduling policies that can generalize across workloads and system configurations.

The framework also includes a comprehensive multi-episode benchmarking pipeline, enabling systematic evaluation of scheduling strategies over varied scenarios.
Results and performance metrics can be exported in multiple formats, including CSV for analysis, PNG for visualization, and JSONL for structured 
logging and reproducibility.

Overall, this project provides a unified platform combining constraint-based optimization and reinforcement learning to address 
next-generation scheduling challenges in heterogeneous high-performance computing systems.


For now, it is a draft that is evolving as it progresses...


















