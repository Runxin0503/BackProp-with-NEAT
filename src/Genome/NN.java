package Genome;


import Evolution.Constants;
import Genome.enums.nodeType;

import java.util.*;

/** Neural Network that uses topological sort to minimize computational time and increase memory efficiency */
public class NN {

    /**
     * Stores a list of edges.
     * <br>Cannot ever decrease in size or have any genome removed
     * <br>Class Invariant: Sorted in InnovationID order
     */
    final ArrayList<edge> genome;

    /**
     * Stores a list of nodes.
     * <br>Cannot ever decrease in size or have any nodes removed.
     * <br>Class Invariant: Sorted in Topological order
     */
    final ArrayList<node> nodes;

    private NN(ArrayList<node> nodes){
        this.nodes = nodes;
        this.genome = new ArrayList<>();
    }

    private NN(ArrayList<node> nodes,ArrayList<edge> genome) {
        this.genome = genome;
        this.nodes = nodes;
    }

    public static NN getDefaultNode() {
        ArrayList<node> nodes = new ArrayList<>();

        for(int i = -Constants.inputNum - Constants.outputNum-1; i<-Constants.outputNum;i++)
            nodes.add(new node(nodeType.input,i,Constants.hiddenAF));
        for(int i = -Constants.outputNum-1; i<0;i++)
            nodes.add(new node(nodeType.output,i,Constants.hiddenAF));

        return new NN(nodes);
    }

    /** Calculates the weighted output of the values using the Neural Network currently in this Agent */
    public double[] calculateWeightedOutput(double[] input) {
        assert input.length < nodes.size();
        double[] calculator = new double[nodes.size()];
        Arrays.fill(calculator,Double.NaN);
        System.arraycopy(input, 0, calculator, 0, input.length);

        for(int i = 0; i < calculator.length; i++) {
            if(Double.isNaN(calculator[i])) continue;
            calculator[i] = nodes.get(i).calculateOutput(calculator[i]);
            for(int j : nodes.get(i).getOutgoingEdgeIndices()){
                edge e = genome.get(j);
                if(e.isDisabled()) continue;
                int targetIndex = e.getNextIndex();
                calculator[targetIndex] += e.calculateOutput(calculator[i]);
            }
        }

        double[] output = new double[Constants.outputNum];
        for(int i=calculator.length-output.length;i<calculator.length;i++){
            if(Double.isNaN(calculator[i])) continue;
            output[i] = calculator[i];
        }
        Constants.outputAF.evaluate(output);
        return output;
    }

    /**
     * Returns a positive value denoting how similar the Genomes of both Neural Networks are<br>
     * Should be the same for if either Neural Network ran this method on each other<br>
     * EX: genome n1,n2; n1.compare(n2) == n2.compare(n1);
     */
    public double compare(NN other){
        if(this.genome.isEmpty() && other.genome.isEmpty()) return 0;
        List<edge> maxInnoGenome = this.genome;
        List<edge> minInnoGenome = other.genome;
        if(maxInnoGenome.isEmpty() || (!minInnoGenome.isEmpty() && minInnoGenome.getLast().getInnovationID() > maxInnoGenome.getLast().getInnovationID())) {
            maxInnoGenome = other.genome;
            minInnoGenome = this.genome;
        }

        int index1 = 0,index2 = 0;

        int disjoint = 0,excess,similar = 0;
        double weight_diff = 0;


        while(index1 < maxInnoGenome.size() && index2 < minInnoGenome.size()){

            edge gene1 = maxInnoGenome.get(index1);
            edge gene2 = minInnoGenome.get(index2);

            int firstInnovationID = gene1.getInnovationID();
            int secondInnovationID = gene2.getInnovationID();

            if(firstInnovationID == secondInnovationID){
                //similargene
                similar ++;
                weight_diff += Math.abs(gene1.getWeight() - gene2.getWeight());
                index1++;
                index2++;
            }else if(firstInnovationID > secondInnovationID){
                //disjoint gene of b
                disjoint ++;
                index2++;
            }else{
                //disjoint gene of a
                disjoint ++;
                index1 ++;
            }
        }

        weight_diff /= Math.max(1,similar);
        excess = maxInnoGenome.size() - index1;

        double N = Math.max(maxInnoGenome.size(), minInnoGenome.size());
        if(N < 20) N = 1;

        return Constants.weightedDisjoints * disjoint / N + Constants.weightedExcess * excess / N + Constants.weightedWeights * weight_diff;

    }

    /**
     * Returns a new Neural Network as a result of crossover between the two given NNs<br>
     * Follows the NEAT Implementation of Innovation number matching
     */
    public static NN crossover(NN parent1, NN parent2, double firstScore, double secondScore){
        int index1 = 0,index2 = 0;
        boolean equalScore = firstScore==secondScore;
        NN dominant = parent1,submissive = parent2;
        if(firstScore<secondScore){
            dominant = parent2;
            submissive = parent1;
        }

        ArrayList<edge> newGenome = new ArrayList<>();
        while(index1 < dominant.genome.size() && index2 < submissive.genome.size()){

            edge gene1 = dominant.genome.get(index1);
            edge gene2 = submissive.genome.get(index2);

            int firstInnovationID = gene1.getInnovationID();
            int secondInnovationID = gene2.getInnovationID();

            if(firstInnovationID == secondInnovationID){
                newGenome.add(Math.random()>0.5 ? gene1.clone(dominant.nodes) : gene2.clone(submissive.nodes));
                index1++;
                index2++;
            }else if(firstInnovationID > secondInnovationID){
                if(equalScore) newGenome.add(gene2.clone(submissive.nodes));
                //disjoint gene of b
                index2++;
            }else{
                //disjoint gene of a
                newGenome.add(gene1.clone(dominant.nodes));
                index1++;
            }
        }

        while(index1 < dominant.genome.size()){
            edge gene1 = dominant.genome.get(index1);
            //if this loop is being run, dominant's last innovationID is greater than submissive's
            //so newGenome shouldn't have gene1 from dominant, and newGenome shouldn't be in submissive since
            //submissive's max innovationID should be less than gene1's innovationID
            assert !newGenome.contains(gene1);
            newGenome.add(gene1.clone(dominant.nodes));
            index1++;
        }
        if(equalScore){
            while(index2 < submissive.genome.size()){
                edge gene2 = submissive.genome.get(index2);
                //same logic as above
                assert !newGenome.contains(gene2);
                newGenome.add(gene2.clone(submissive.nodes));
                index2++;
            }
        }
        //newGenome must be sorted by Innovation ID since they were inserted in order from smallest to biggest
        List<edge> test = new ArrayList<>(newGenome);
        test.sort(Comparator.comparingInt(edge::getInnovationID));
        assert test.equals(newGenome);

        ArrayList<node> newNodes = Innovation.constructNetworkFromGenome(newGenome,dominant.nodes,submissive.nodes);
        
        return new NN(newNodes,newGenome);
    }

    /** Randomly mutates this Neural Network */
    public void mutate(){
        if(!genome.isEmpty()) {
            if (Math.random() < Constants.mutationWeightShiftProbability) Mutation.shiftWeights(this);
            if (Math.random() < Constants.mutationWeightRandomProbability) Mutation.randomWeights(this);
            if (Math.random() < Constants.mutationNodeProbability) Mutation.mutateNode(this);
        }
        if(Math.random() < Constants.mutationBiasShiftProbability) Mutation.shiftBias(this);
        if(Math.random() < Constants.mutationSynapseProbability) Mutation.mutateSynapse(this);
    }
}
