package Evolution;

import Genome.Genome;
import globalGenomes.globalInnovations;
import globalGenomes.globalNodes;

import java.util.ArrayList;
import java.util.Random;

public class Evolution {
    private final globalInnovations globalInnovations;
    private final globalNodes globalNodes;
    private final ArrayList<Species> Speciation;
    public final ArrayList<NN> NeuralNets;
    private final int numSimulated;
    private final double perctCull;
    public static int inputNum,outputNum,maxStagDropoff;
    public static String hiddenAF,outputAF;
    public static double compatibilityThreshold,weightedExcess,weightedDisjoints,weightedWeights;
    public static double mutationSynapseProbability,mutationNodeProbability,mutationWeightShiftProbability,mutationWeightRandomProbability,mutationBiasShiftProbability,mutationWeightShiftStrength,mutationWeightRandomStrength,mutationBiasShiftStrength;
    private Random rand;

    public static void main(String args[]){
        Evolution evo = new Evolution(1000);
    }
    public Evolution(int numSimulated){
        //config
        inputNum=4;
        outputNum=1;

        hiddenAF="sigmoid";
        outputAF="none";

        weightedExcess = 1;
        weightedDisjoints = 1;
        weightedWeights = 1;
        compatibilityThreshold = 4;
        maxStagDropoff = 20;

        mutationSynapseProbability=0.01;//*3
        mutationNodeProbability=0.1;//*2

        mutationWeightShiftProbability=0.02;//*3
        mutationWeightRandomProbability=0.02;//*3
        mutationBiasShiftProbability=0.02;//*3

        mutationWeightShiftStrength=1;//2
        mutationWeightRandomStrength=2;
        mutationBiasShiftStrength=0.3;

        this.perctCull = 0.2;
        this.NeuralNets = new ArrayList<NN>();
        this.Speciation = new ArrayList<Species>();
        this.numSimulated = numSimulated;
        this.globalNodes = new globalNodes(inputNum,outputNum);
        this.globalInnovations = new globalInnovations(globalNodes);

        NN first = new NN(new Genome(globalInnovations,globalNodes));
        NeuralNets.add(first);
        Speciation.add(new Species(first));
        for(int i=1;i<numSimulated;i++){
            NN temp = new NN(new Genome(globalInnovations,globalNodes));
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

        for(NN nn : NeuralNets){
            boolean found = false;
            for(Species s : Speciation){
                if(s.add(nn)){
                    found=true;
                    break;
                }
            }
            if(!found)Speciation.add(new Species(nn));
        }

        //update stagnant count, then cull
        for(Species s : Speciation){
            s.updateStag();
            s.cull(perctCull);
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
        for(NN nn : NeuralNets){
            if(nn.genome==null){
                pickSpecies(populationScore).populateGenome(nn);
            }
        }

        //mutate
        for(NN nn : NeuralNets){
            nn.genome.mutate();
        }

        //reset
        for(NN nn:NeuralNets){
            nn.reset();
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
