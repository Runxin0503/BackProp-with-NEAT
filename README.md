# BackProp with NEAT

This project is a **comprehensive implementation of the NEAT (NeuroEvolution of Augmenting Topologies)** algorithm, written entirely from scratch in Java. In addition to the traditional evolutionary capabilities of NEAT, this implementation incorporates **backpropagation** and the **Adam optimizer**, enabling hybrid learning by combining neuroevolution with gradient-based optimization.

---

## Features

- **Core NEAT Functionality**:
  - Evolving neural network topologies and weights simultaneously.
  - Speciation mechanism to preserve diversity in the population.
  - Crossover and mutation operators for topological innovation.
  - Compatibility scoring for speciation.

- **Backpropagation**:
  - Supports gradient-based weight updates within evolved networks.
  - Compatible with various Activation functions (ReLU, sigmoid, tanh, etc.).
  - Can support custom Activation functions by manually adding them in the activation enum file

- **Adam Optimizer**:
  - Efficient gradient-based optimization with adaptive learning rates.
  - Handles sparse gradients and accelerates convergence.

- **Highly Modular Design**:
  - Easily extendable classes for custom mutation rates, fitness functions, and network topologies.
  - Designed for flexibility in hybrid training strategies (e.g., evolution followed by fine-tuning).

## How It Works
- ### **NEAT Algorithm**:
- **Population Initialization**: Creates a population of simple neural networks.
- **Fitness Evaluation**: Evaluates how well each genome solves the task.
- **Speciation**: Groups similar genomes into species to preserve diversity.
- **Selection and Reproduction**: Select the best genomes for crossover and mutation.
- **Innovation Tracking**: Tracks innovations to align structures during crossover.
- **Backpropagation and Adam Optimizer**: Before training starts, the Agent's genome is cloned and backpropagation is applied to the cloned genome during training. This approach allows the genome to avoid getting stuck in local minima and enhances the accuracy of evolved networks without sacrificing the benefits of topology evolution

---
