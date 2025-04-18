package Evolution;

import Genome.Activation;
import Genome.Cost;
import Genome.Optimizer;
import Genome.NN;

import java.util.ArrayList;
import java.util.function.Function;

/** TODO */
public class Evolution {

    /** The collection of {@linkplain Species} used in the NEAT algorithm. */
    private final ArrayList<Species> species = new ArrayList<>();

    /** The collection of {@linkplain Agent} classes, whose Genome is selected, trained, and
     * improved according to the NEAT algorithm. */
    public final Agent[] agents;

    /** The Constants class for all subsequent dependent classes like {@linkplain NN} and {@linkplain Species}. */
    public Constants Constants;

    private Evolution(Constants Constants, Function<Integer, ? extends Agent> agentConstructor) {
        this.Constants = Constants;
        Agent first = agentConstructor.apply(0);
        agents = new Agent[Constants.numSimulated];
        agents[0] = first;
        species.add(new Species(first, Constants));
        for (int i = 1; i < Constants.numSimulated; i++) {
            Agent temp = agentConstructor.apply(i);
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
            if (!found) species.add(new Species(agent, Constants));
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
        private final Constants Constants = new Constants();
        private int initialMutation = 10;
        private Function<Integer, Agent> agentConstructor = i -> new Agent(Constants, initialMutation);

        /** TODO */
        public EvolutionBuilder setNumSimulated(int numSimulated) {
            Constants.numSimulated = numSimulated;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setInputNum(int inputNum) {
            Constants.inputNum = inputNum;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setOutputNum(int outputNum) {
            Constants.outputNum = outputNum;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setInitialMutation(int initialMutation) {
            this.initialMutation = initialMutation;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setOutputAF(Activation.arrays outputAF) {
            Constants.outputAF = outputAF;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setDefaultHiddenAF(Activation defaultHiddenAF) {
            Constants.defaultHiddenAF = defaultHiddenAF;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setCostFunction(Cost CostFunction) {
            Constants.CostFunction = CostFunction;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setAgentConstructor(Function<Integer, Agent> agentConstructor) {
            this.agentConstructor = agentConstructor;
            return this;
        }

        /** TODO */
        public EvolutionBuilder setOptimizer(Optimizer optimizer) {
            this.Constants.optimizer = optimizer;
            return this;
        }

        /** TODO */
        public Evolution build() throws MissingInformation {
            if (Constants.inputNum == -1 || Constants.outputNum == -1 || Constants.numSimulated == -1 || Constants.outputAF == null || Constants.CostFunction == null)
                throw new MissingInformation();
            Constants.defaultValueInitializer = Activation.getInitializer(Constants.defaultHiddenAF, Constants.inputNum, Constants.outputNum);
            return new Evolution(Constants, agentConstructor);
        }

        /** TODO */
        public static class MissingInformation extends RuntimeException {
            @Override
            public String getMessage() {
                return "Missing Resources in EvolutionBuilder class";
            }
        }
    }
}
