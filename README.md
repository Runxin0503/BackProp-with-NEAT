# BackProp with NEAT

This project is a **comprehensive implementation of the NEAT (NeuroEvolution of Augmenting Topologies)** algorithm, written entirely from scratch in Java. In addition to the traditional evolutionary capabilities of NEAT, this implementation incorporates **backpropagation** and the **Adam optimizer**, enabling hybrid learning by combining neuroevolution with gradient-based optimization.

## Features

- **Core NEAT Functionality**:
  - Evolving neural network topologies and weights simultaneously.
  - Speciation mechanism to preserve diversity in the population.
  - Crossover and mutation operators for topological innovation.
  - Compatibility scoring for speciation.

- **Backpropagation**:
  - Supports gradient-based weight updates within evolved networks.
  - Compatible with various Activation functions (ReLU, sigmoid, tanh, etc.).
  - Can support custom Activation functions by manually adding them in the activation enum file.

- **Adam Optimizer**:
  - Efficient gradient-based optimization with adaptive learning rates.
  - Handles sparse gradients and accelerates convergence.

- **Highly Modular Design**:
  - Easily extendable classes for custom mutation rates, fitness functions, and network topologies.
  - Designed for flexibility in hybrid training strategies (e.g., evolution followed by fine-tuning).

## How It Works

### **NEAT Algorithm**:
- **Population Initialization**: Creates a population of simple neural networks.
- **Fitness Evaluation**: Evaluates how well each genome solves the task.
- **Speciation**: Groups similar genomes into species to preserve diversity.
- **Selection and Reproduction**: Select the best genomes for crossover and mutation.
- **Innovation Tracking**: Tracks innovations to align structures during crossover.
- **Backpropagation and Adam Optimizer**: Before training starts, the Agent's genome is cloned and backpropagation is applied to the cloned genome during training. This approach allows the genome to avoid getting stuck in local minima and enhances the accuracy of evolved networks without sacrificing the benefits of topology evolution.

## Classes Overview

### `Evolution` Class:
- The core class implementing the NEAT algorithm. It manages a population of `Agent`s, grouped by `Species`, and evolves their neural network genomes using selection, crossover, mutation, and speciation strategies.

  - **Key Methods**:
    - `nextGen()`: Evolve the population by applying the NEAT algorithm after each generation is evaluated. The `Agent` and `Species` classes can be overridden to create custom implementations of agents and species within this stochastic evolutionary process.

### `Agent` Class:
- Represents an individual in the population, containing a neural network genome that evolves over generations. Each agent's performance is evaluated with a score, which is used for selection in the evolutionary process. This class can be overridden to implement custom agent behaviors or specialized fitness evaluations.

  - **Key Methods**:
    - `reset()`: Resets the agent's score.
    - `mutate()`: Mutates the agent's genome.
    - `crossover()`: Combines the genomes of two agents.
    - `getGenomeClone()`: Returns a clone of the agent's genome for further processing like backpropagation.

### `Species` Class:
- Groups agents based on the similarity of their genomes, ensuring that genetic diversity is preserved. The `Species` class can also be customized to define different behaviors for how agents are grouped or how species interact with each other during evolution.

## How to Run

1. Clone this repository.
2. Install [Maven](https://maven.apache.org/install.html) (if not already installed).
3. Navigate to the project directory and run:
   ```bash
   mvn clean install
4. Use the provided classes in your project or run the Evolution class to begin experimenting with the NEAT algorithm.

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.