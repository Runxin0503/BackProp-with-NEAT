package Evolution;

import Genome.NN;
import Genome.Renderer;

/**
 * Represents an individual in the NEAT (NeuroEvolution of Augmenting Topologies) algorithm population.
 * <p>
 * Each {@code Agent} contains a neural network genome, which it evolves through mutation and crossover operations.
 * The agent's performance is evaluated using a {@code score}, which is used for selection in the evolutionary process.
 * </p>
 *
 * <p>Agents are managed by the {@link Evolution} class and grouped in {@link Species} for speciation-based evolution.</p>
 *
 * @see NN
 * @see Renderer
 * @see Evolution
 * @see Species
 */
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

    protected Agent(Constants Constants, int initialMutation) {
        this.score = 0;
        this.genome = NN.getDefaultNeuralNet(Constants);
        for (int i = 0; i < initialMutation; i++) genome.mutate();
    }

    /** Resets the score of this Agent.<br>
     * Called at the end of {@link Evolution#nextGen()} right after {@link Agent#mutate()}.*/
    public void reset() {
        score = 0;
    }

    /** Returns the score of this Agent as input to the NEAT algorithm.<br>
     * Called multiple times throughout {@link Evolution#nextGen()} to get
     * the performance of this Agent's Genome. */
    @Override
    public double getScore() {
        return score;
    }

    /** Sets the score of this Agent.<br>
     * Should be called before {@link Evolution#nextGen()} to set Agent's
     * score relative to its Genome's performance. */
    public void setScore(double newScore) {
        if (Double.isNaN(newScore)) throw new RuntimeException("Attempt to set invalid score {" + newScore + "}");
        score = Math.max(0, newScore);
    }

    /** Returns a clone of this Agent's Genome, can be used for rendering
     * in {@link Renderer}, backpropagation evaluation in {@link NN#learn}, or
     * many more in the {@link NN} class.
     * @throws NullPointerException if this Agent doesn't have a Genome */
    public NN getGenomeClone() {
        if (!hasGenome()) throw new NullPointerException("Agent " + this + " has empty Genome");
        return (NN) genome.clone();
    }

    /** Removes the genome of this Agent.<br>
     * Called during {@link Evolution#nextGen()} in {@link Species#cull()} to remove Genomes with bad
     * performance from the population.
     * @throws NullPointerException if this Agent doesn't have a Genome */
    protected void removeGenome() {
        if(!hasGenome()) throw new NullPointerException("Agent " + this + " has empty Genome");
        genome = null;
    }

    /**
     * Replaces {@code child} Agent's genome with the crossover result of {@code this} and {@code otherParent}.
     * <br>Called during {@link Evolution#nextGen()} in {@link Species#populateGenome} to repopulate Agents with
     * well-performing Genomes after bad-performing Genomes are removed from the population in {@link Species#cull()}.
     * @throws NullPointerException if either parent are missing their genome.
     * @throws RuntimeException if child already has a genome.
     */
    protected void crossover(Agent otherParent, Agent child) {
        if (!this.hasGenome())
            throw new NullPointerException("Parent Agent " + this + " has empty Genome");
        else if(!otherParent.hasGenome())
            throw new NullPointerException("Parent Agent " + otherParent + " has empty Genome");
        else if(child.hasGenome())
            throw new RuntimeException("Child Agent " + child + " already has Genome");
        child.genome = NN.crossover(this.genome, otherParent.genome, this.score, otherParent.score);
    }

    /** Returns whether this Agent has a genome or not */
    public boolean hasGenome() {
        return genome != null;
    }

    /** Mutates the genome of this Agent.<br>
     * Called in {@link Evolution#nextGen()} after {@link Species#populateGenome} to mutate
     * all existing Genomes in all Agents.
     * @throws RuntimeException when either Agent is missing a genome.
     */
    protected void mutate() {
        if(!hasGenome()) throw new NullPointerException("Agent " + this + " has empty Genome");
        genome.mutate();
    }

    /**
     * Compares the genome of both Agents.
     * @return the value of the comparison.
     * @throws RuntimeException when either Agent is missing a genome.
     */
    protected double compare(Agent newAgent) {
        if (!hasGenome() || !newAgent.hasGenome()) throw new RuntimeException("Genome Exception");
        return NN.compare(genome, newAgent.genome);
    }

    @Override
    public String toString() {
        return genome.toString();
    }
}