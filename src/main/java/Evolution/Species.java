package Evolution;

import java.util.ArrayList;

/**
 * Used in a part of the NEAT algorithm to determine if a genome is successful or not based on its species
 * <br>Randomly selects a representative from its population every generation to use as comparison during
 * species identification.
 */
public class Species implements WeightedRandom {
    /** The representative of this species. Used during species identification and randomly chosen every new generation */
    private Agent representative;

    /** Represents how stagnant this species is at improving their {@link #populationScore}.
     * <br><br> -Increases by 1 every time the current generation does worse than the previous
     * <br><br> -Decreases by a factor of the improvement when the current generation does than the previous. */
    private int stag = 0;

    /** The total score of all agents within this species */
    private double populationScore;

    /** Arraylist containing all members of this species */
    private final ArrayList<Agent> population = new ArrayList<Agent>();

    /** The Constants object for the {@linkplain Evolution} class that created this class instance. */
    private final Constants Constants;

    protected Species(Agent representative, Constants Constants) {
        this.representative = representative;
        this.population.add(representative);
        this.populationScore = representative.getScore();
        this.Constants = Constants;
    }

    /**
     * Attempts to add {@code newAgent} to this species if it's genome is similar enough.<br>
     * Called in {@link Evolution#nextGen()} when re-assigning every Agent to their appropriate Species.
     * @return true if successful, false otherwise
     */
    boolean add(Agent newAgent) {
        if (representative.compare(newAgent) < Constants.compatibilityThreshold) {

            //keeps score sorted from highest to lowest
            if (population.isEmpty() || population.getLast().getScore() > newAgent.getScore()) {
                population.add(newAgent);
            } else {
                for (int i = 0; i < population.size(); i++) {
                    if (population.get(i).getScore() < newAgent.getScore()) {
                        population.add(i, newAgent);
                        break;
                    }
                }
            }
            return true;
        }
        return false;
    }

    /** Calculates the {@link #populationScore} of all members of this species, taking into account {@link #stag}. */
    protected void calculateScore() {
        populationScore = 0;
        for (Agent agent : population) {
            populationScore += agent.getScore();
        }
        populationScore /= population.size();
        if (stag >= Constants.maxStagDropoff) populationScore *= 0.7;
    }

    /** Updates the {@link #stag} of this species. */
    protected void updateStag() {
        double count = 0;
        for (Agent agent : population) {
            count += agent.getScore();
        }
        count /= population.size();
        if (populationScore > count) stag++;
        else stag = Math.max(0, (int) Math.round(stag * (1 - (count - populationScore) / populationScore)));
    }

    /**
     * Randomly chooses a {@link #representative} from all members of this species and
     * then clears the {@link #population} arraylist.
     */
    protected void reset() {
        representative = population.get((int) (Math.random() * population.size()));
        population.clear();
    }

    /**
     * Effect: A percentage ({@linkplain Constants#perctCull}) of the worst performing members
     * of this Species have their Genome removed through {@link Agent#removeGenome()}.
     */
    protected void cull() {
        int numSurvived = (int) (Math.round(population.size() * (1 - Constants.perctCull)));
        for (int i = population.size() - 1; i > numSurvived; i--) {
            population.remove(i).removeGenome();
        }
    }

    /** Returns if this species has no current members. */
    protected boolean isEmpty() {
        return population.isEmpty();
    }

    /** Randomly selects two Agents from this Species (weighted by {@link Agent#getScore()})
     *  to repopulate {@code emptyAgent}'s Genome.<br>
     * Called in {@link Evolution#nextGen} after {@link #cull} on the Agents with empty Genomes. */
    protected void populateGenome(Agent emptyAgent) {
        Agent first = WeightedRandom.getRandom(population);
        Agent second = WeightedRandom.getRandom(population);
        first.crossover(second, emptyAgent);
        population.add(emptyAgent);
    }

    /** Returns the total score of all Agents in this Species as input to the NEAT algorithm.<br>
     * Called multiple times throughout {@link Evolution#nextGen()} to get
     * the performance of this Species. */
    @Override
    public double getScore() {
        return populationScore;
    }
}
