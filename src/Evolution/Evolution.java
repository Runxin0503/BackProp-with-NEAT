package Evolution;

import java.util.ArrayList;
import java.util.Random;

public class Evolution {
    private final ArrayList<Species> Speciation = new ArrayList<>();
    public final ArrayList<Agent> NeuralNets = new ArrayList<>();
    private final int numSimulated;
    private Random rand;

    public Evolution(int numSimulated){
        this.numSimulated = numSimulated;

        Agent first = new Agent();
        NeuralNets.add(first);
        Speciation.add(new Species(first));
        for(int i=1;i<numSimulated;i++){
            Agent temp = new Agent();
            Speciation.get(0).add(temp);
            NeuralNets.add(temp);
        }
    }

    public void nextGen(){
//        System.out.println(Speciation.get(0).NeuralNets.get(0));
        //Species Separation
        for(Species s : Speciation){
            s.reset();
        }

        for(Agent agent : NeuralNets){
            boolean found = false;
            for(Species s : Speciation){
                if(s.add(agent)){
                    found=true;
                    break;
                }
            }
            if(!found)Speciation.add(new Species(agent));
        }

        //update stagnant count, then cull
        for(Species s : Speciation){
            s.updateStag();
            s.cull(Constants.perctCull);
        }

        //remove extinct
        for(int i=Speciation.size()-1;i>=0;i--){
            Species s = Speciation.get(i);
            if(s.NeuralNets.isEmpty()){
                s.extinct();
                Speciation.remove(i);
            }
        }

        for(Species s : Speciation) s.calculateScore();

        //repopulate Genomes & reproduce
        double populationScore = 0;
        for(Species s : Speciation)populationScore += s.speciesScore;
        for(Agent agent : NeuralNets){
            if(agent.NN ==null){
                pickSpecies(populationScore).populateGenome(agent);
            }
        }

        //mutate
        for(Agent agent : NeuralNets){
            agent.NN.mutate();
        }

        //reset
        for(Agent agent :NeuralNets){
            agent.reset();
        }
        System.out.println("Species ("+Speciation.size()+")");
    }

    private Species pickSpecies(double populationScore){
        if(populationScore==0)return Speciation.get((int)(Math.random() * Speciation.size()));
        double random = Math.random() * populationScore;
        for(Species s : Speciation){
            random -= s.speciesScore;
            if(random<=0){
                return s;
            }
        }
        return Speciation.get((int)(Math.random() * Speciation.size()));
    }
}
