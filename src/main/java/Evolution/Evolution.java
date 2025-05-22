package Evolution;

import Genome.Activation;
import Genome.Cost;
import Genome.NN;
import Genome.Optimizer;

import java.util.ArrayList;
import java.util.function.BiFunction;

/**
 * The core class implementing the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.
 * <p>
 * {@code Evolution} manages a population of {@link Agent}s, grouped by {@link Species}, and evolves their
 * neural network genomes using selection, crossover, mutation, and speciation strategies.
 * </p>
 *
 * <p>Call {@link #nextGen()} after each generation is evaluated to evolve the population.</p>
 *
 * @see Agent
 * @see Species
 * @see EvolutionBuilder
 */
public class Evolution {

    /** The collection of {@linkplain Species} used in the NEAT algorithm. */
    private final ArrayList<Species> species = new ArrayList<>();

    /** The collection of {@linkplain Agent} classes, whose Genome is selected, trained, and
     * improved according to the NEAT algorithm. */
    public final Agent[] agents;

    /** The Constants class for all subsequent dependent classes like {@linkplain NN} and {@linkplain Species}. */
    public Constants Constants;

    /** A custom Species constructor function used to create subclasses of the {@linkplain Species} class. */
    private final BiFunction<Agent, Constants, Species> speciesConstructor;

    private Evolution(Constants Constants,
                      BiFunction<Integer, Constants, Agent> agentConstructor,
                      BiFunction<Agent, Constants, Species> speciesConstructor, int initialMutation) {
        this.Constants = Constants;
        this.speciesConstructor = speciesConstructor;

        Agent first = agentConstructor.apply(initialMutation, Constants);
        agents = new Agent[Constants.getNumSimulated()];
        agents[0] = first;
        species.add(speciesConstructor.apply(first, Constants));
        for (int i = 1; i < Constants.getNumSimulated(); i++) {
            Agent temp = agentConstructor.apply(initialMutation, Constants);
            species.getFirst().add(temp);
            agents[i] = temp;
        }
    }

    /** Once every agent has a valid score, {@code nextGen} applies the NEAT algorithm on the Agents and create the next generation */
    public void nextGen() {
        double populationScore = 0;
        for (Agent agent : agents) {
            populationScore += agent.getScore();
        }
        System.out.println("Generation score: " + populationScore);

        //Species Separation
        for (Species s : species) {
            s.reset();
        }

        //assign all agents to a species
        for (Agent agent : agents) {
            boolean found = false;
            for (Species s : species) {
                if (s.add(agent)) {
                    found = true;
                    break;
                }
            }
            if (!found) species.add(speciesConstructor.apply(agent, Constants));
        }

        //update stagnant count and then cull
        for (Species s : species) {
            s.updateStag();
            s.cull();
        }

        //remove empty species
        for (int i = species.size() - 1; i >= 0; i--) {
            Species s = species.get(i);
            if (s.isEmpty()) species.remove(i);

        }

        //calculates population score
        for (Species s : species) s.calculateScore();

        //repopulate Genomes & reproduce
        for (Agent agent : agents) {
            if (!agent.hasGenome()) {
                WeightedRandom.getRandom(species).populateGenome(agent);
            }
        }

        //mutate Genome and reset score
        for (Agent agent : agents) {
            agent.mutate();
            agent.reset();
        }
    }

    /**
     * The builder class for {@link Evolution}, a factory that produces, trains, and applies the NEAT genetic algorithm on neural network agents.
     */
    public static class EvolutionBuilder {
        private int numSimulated = -1;
        private int inputNum = -1;
        private int outputNum = -1;
        private Activation.arrays outputAF = null;
        private Activation defaultHiddenAF = Activation.none;
        private Cost CostFunction = null;
        private Optimizer optimizer = Optimizer.ADAM;
        private int initialMutation = 10;
        private BiFunction<Integer, Constants, Agent> agentConstructor = (initialMutation, Constants) -> new Agent(Constants, initialMutation);
        private BiFunction<Agent, Constants, Species> speciesConstructor = (representative, Constants) -> new Species(representative, Constants);

        /**
         * Sets the number of agents to simulate per generation.
         *
         * @param numSimulated number of agents in the population
         * @return the current builder instance
         */
        public EvolutionBuilder setNumSimulated(int numSimulated) {
            this.numSimulated = numSimulated;
            return this;
        }

        /**
         * Sets the number of input nodes for each agent's neural network.
         *
         * @param inputNum number of input neurons
         * @return the current builder instance
         */
        public EvolutionBuilder setInputNum(int inputNum) {
            this.inputNum = inputNum;
            return this;
        }

        /**
         * Sets the number of output nodes for each agent's neural network.
         *
         * @param outputNum number of output neurons
         * @return the current builder instance
         */
        public EvolutionBuilder setOutputNum(int outputNum) {
            this.outputNum = outputNum;
            return this;
        }

        /**
         * Sets how many times each agent's genome is initially mutated upon creation.
         *
         * @param initialMutation number of initial mutation calls
         * @return the current builder instance
         */
        public EvolutionBuilder setInitialMutation(int initialMutation) {
            this.initialMutation = initialMutation;
            return this;
        }

        /**
         * Sets the activation function used at the output layer of the neural networks.
         *
         * @param outputAF activation function for the output layer
         * @return the current builder instance
         */
        public EvolutionBuilder setOutputAF(Activation.arrays outputAF) {
            this.outputAF = outputAF;
            return this;
        }

        /**
         * Sets the default activation function used in hidden layers of the neural networks.
         *
         * @param defaultHiddenAF the default activation function
         * @return the current builder instance
         */
        public EvolutionBuilder setDefaultHiddenAF(Activation defaultHiddenAF) {
            this.defaultHiddenAF = defaultHiddenAF;
            return this;
        }

        /**
         * Sets the cost/loss function used to evaluate neural network performance.
         * <br>This set function can be ignored. Defaults to {@code null} and disables
         * {@linkplain NN#calculateCost} and {@linkplain NN#learn}.
         * @param CostFunction the cost function
         * @return the current builder instance
         */
        public EvolutionBuilder setCostFunction(Cost CostFunction) {
            this.CostFunction = CostFunction;
            return this;
        }

        /**
         * Sets a custom agent constructor function.
         * <p>Useful for supplying subclasses or injecting behavior into agent creation.</p>
         *
         * @param agentConstructor a function taking an integer index and returning an Agent
         * @return the current builder instance
         */
        public EvolutionBuilder setAgentConstructor(BiFunction<Integer, Constants, Agent> agentConstructor) {
            this.agentConstructor = agentConstructor;
            return this;
        }

        /**
         * Sets a custom Species constructor function.
         * <p>Useful for supplying subclasses or injecting behavior into Species creation.</p>
         *
         * @param speciesConstructor a supplier that supplies Species instance classes
         * @return the current builder instance
         */
        public EvolutionBuilder setSpeciesConstructor(BiFunction<Agent, Constants, Species> speciesConstructor) {
            this.speciesConstructor = speciesConstructor;
            return this;
        }

        /**
         * Sets the optimizer used during backpropagation learning (if applicable).
         *
         * @param optimizer the optimizer to use
         * @return the current builder instance
         */
        public EvolutionBuilder setOptimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        /**
         * Builds the {@link Evolution} object after validating that all required parameters are present.
         *
         * @return a fully constructed {@link Evolution} instance
         * @throws MissingInformation if any required configuration field is missing
         */
        public Evolution build() throws MissingInformation {
            if (inputNum == -1 || outputNum == -1 || numSimulated == -1 || outputAF == null)
                throw new MissingInformation();
            Constants Constants = new Constants(inputNum, outputNum, numSimulated, defaultHiddenAF, outputAF, CostFunction, optimizer);
            return new Evolution(Constants, agentConstructor, speciesConstructor, initialMutation);
        }

        /**
         * Builds the {@link Evolution} object with the provided {@code Constants} class.
         *
         * @return a fully constructed {@link Evolution} instance
         */
        public Evolution buildWithConstants(Constants Constants) {
            return new Evolution(Constants, agentConstructor, speciesConstructor, initialMutation);
        }

        /**
         * Exception thrown when required configuration fields are missing during {@link #build()}.
         */
        public static class MissingInformation extends Exception {
            @Override
            public String getMessage() {
                return "Missing Resources in EvolutionBuilder class";
            }
        }
    }
}
