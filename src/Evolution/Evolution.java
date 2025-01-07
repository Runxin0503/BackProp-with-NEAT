package Evolution;

import Genome.enums.Activation;

import java.util.*;

public class Evolution {
    private final ArrayList<Species> species = new ArrayList<>();
    public final Agent[] agents;

    public Constants Constants;

    private Evolution(Constants Constants){
        this.Constants = Constants;
        Agent first = new Agent(Constants);
        agents = new Agent[Constants.numSimulated];
        agents[0] = first;
        species.add(new Species(first,Constants));
        for(int i=1;i<Constants.numSimulated;i++){
            Agent temp = new Agent(Constants);
            species.getFirst().add(temp);
            agents[i] = temp;
        }
    }

    public void nextGen(){
        //Species Separation
        for(Species s : species){
            s.reset();
        }

        //assign all agents to a species
        for(Agent agent : agents){
            boolean found = false;
            for(Species s : species){
                if(s.add(agent)){
                    found=true;
                    break;
                }
            }
            if(!found) species.add(new Species(agent,Constants));
        }

        //update stagnant count, then cull
        for(Species s : species){
            s.updateStag();
            s.cull();
        }

        //remove empty species
        for(int i = species.size()-1; i>=0; i--){
            Species s = species.get(i);
            if(s.isEmpty()) species.remove(i);

        }

        //calculates population score
        for(Species s : species) s.calculateScore();

        //repopulate Genomes & reproduce
        for(Agent agent : agents){
            if(!agent.hasGenome()){
                WeightedRandom.getRandom(species).populateGenome(agent);
            }
        }

        //mutate
        for(Agent agent : agents){
            agent.mutate();
        }

        //reset
        for(Agent agent : agents){
            agent.reset();
        }
    }

    /**
     * The builder class for {@link Evolution}, which is a factory that produces, trains, and applies the NEAT genetic algorithm on neural network agents.
     */
    public class EvolutionBuilder {
        private Constants Constants = new Constants();

        public EvolutionBuilder(){}

        public EvolutionBuilder setNumSimulated(int numSimulated){
            Constants.numSimulated = numSimulated;
            return this;
        }

        public EvolutionBuilder setInputNum(int inputNum){
            Constants.inputNum = inputNum;
            return this;
        }

        public EvolutionBuilder setOutputNum(int outputNum){
            Constants.outputNum = outputNum;
            return this;
        }

        public EvolutionBuilder setOutputAF(Activation.arrays outputAF){
            Constants.outputAF = outputAF;
            return this;
        }

        public EvolutionBuilder setDefaultHiddenAF(Activation defaultHiddenAF){
            Constants.defaultHiddenAF = defaultHiddenAF;
            return this;
        }

        public Evolution build() throws MissingInformation {
            if(Constants.inputNum==-1 || Constants.outputNum==-1 || Constants.numSimulated==-1 || Constants.outputAF == null) throw new MissingInformation();
            return new Evolution(Constants);
        }

        private static class MissingInformation extends Exception {
            @Override
            public String getMessage() {
                return "Missing Resources in EvolutionBuilder class";
            }
        }
    }
}
