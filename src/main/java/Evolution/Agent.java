package Evolution;

import Genome.NN;

/** TODO */
public class Agent implements WeightedRandom {

    /**
     * The genome of this Agent
     * <br> Can be null if this Agent's genome is currently being repurposed by a better-performing genome
     */
    private NN genome;

    /**
     * The score of this Agent.
     * <br> Evaluates the performance of this Agent and its genome.
     */
    private double score;

    public Agent(Constants Constants, int initialMutation) {
        this.score = 0;
        this.genome = NN.getDefaultNeuralNet(Constants);
        for(int i=0;i<initialMutation;i++) genome.mutate();
    }

    /** Resets the score of this Agent
     * TODO add when this method is called in Evolution.nextGen sequence */
    public void reset() {
        score = 0;
    }

    /** Returns the score of this Agent as input to the NEAT algorithm
     * TODO add when this method is called in Evolution.nextGen sequence */
    @Override
    public double getScore() {
        return score;
    }

    /** Sets the score of this Agent
     * TODO add when this method is called in Evolution.nextGen sequence */
    public void setScore(double newScore) {
        if(Double.isNaN(newScore)) throw new RuntimeException("Attempt to set invalid score {"+newScore+"}");
        score = Math.max(0,newScore);
    }

    /** Returns the clone of this Agent's genome, throws a {@link NullPointerException} if this Agent doesn't have a genome
     * TODO add when this method is called in Evolution.nextGen sequence */
    public NN getGenomeClone(){return (NN) genome.clone();}

    /** Removes the genome of this Agent.
     * TODO add when this method is called in Evolution.nextGen sequence */
    protected void removeGenome() {
        assert hasGenome();
        genome = null;
    }

    /**
     * Replaces {@code child} Agent's genome with the crossover result of {@code parent1} and {@code parent2}
     * @throws RuntimeException if parent 1 or 2 are missing their genome or if child already has a genome
     * TODO add when this method is called in Evolution.nextGen sequence
     */
    protected void crossover(Agent otherParent, Agent child) {
        if (!this.hasGenome() || !otherParent.hasGenome() || child.hasGenome())
            throw new RuntimeException("Genome Exception");
        child.genome = NN.crossover(this.genome, otherParent.genome, this.score, otherParent.score);
    }

    /** Returns whether this Agent has a genome or not */
    public boolean hasGenome() {
        return genome != null;
    }

    /** Mutates the genome of this Agent.
     * TODO add when this method is called in Evolution.nextGen sequence */
    protected void mutate() {
        genome.mutate();
    }

    /**
     * Compares the genome of both Agents
     * @return the value of the comparison
     * @throws RuntimeException when either Agent is missing a genome
     */
    protected double compare(Agent newAgent) {
        if (!hasGenome() || !newAgent.hasGenome()) throw new RuntimeException("Genome Exception");
        return NN.compare(genome,newAgent.genome);
    }

    @Override
    public String toString() {
        return genome.toString();
    }
}