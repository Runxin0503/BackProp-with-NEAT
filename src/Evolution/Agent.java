package Evolution;

import Genome.*;

public class Agent implements WeightedRandom{
    /**
     *
     */
    private NN genome;

    /**
     *
     */
    private double score;

    public Agent(){
        this.score=0;
        this.genome = Innovation.getDefaultNode();
    }

    /** Resets the score of this Agent */
    public void reset(){
        score=0;
    }

    public double getScore() {

    }

    public void removeGenome() {
    }

    public NN crossover(Agent first, Agent second) {
    }

    public boolean hasGenome() {
    }

    public void mutate() {
    }

    @Override
    public double getValue() {
        return score;
    }

    /** Calculates the weighted output of the values using the Neural Network currently in this Agent */
    public double[] calculateWeightedOutput(double[] input) {
        return genome.calculateWeightedOutput(input);
    }

    @Override
    public String toString(){
        return genome.toString();
    }

    public double compare(Agent newAgent) {
    }
}