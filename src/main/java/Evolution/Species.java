package Evolution;

import java.util.ArrayList;

/**
 * Used in a part of the NEAT algorithm to determine if a genome is successful or not based on its species
 * <br>Randomly selects a representative from its population every generation to use as comparison during
 * species identification.
 * TODO Make this Overridable in Evolution.
 */
public class Species implements WeightedRandom {
    /** The representative of this species. Used during species identification and randomly chosen every new generation */
    private Agent representative;

    /** Represents how stagnant this species is at improving. The higher the value, the more competitive this species is */
    private int stag = 0;

    /** The total score of all agents within this species */
    private double populationScore;

    /** Arraylist containing all members of this species */
    private final ArrayList<Agent> population = new ArrayList<Agent>();

    /** TODO */
    private final Constants Constants;

    /** TODO */
    protected Species(Agent representative, Constants Constants) {
        this.representative = representative;
        this.population.add(representative);
        this.populationScore = representative.getScore();
        this.Constants = Constants;
    }

    /**
     * Attempts to add {@code newAgent} to this species if it's genome is similar enough
     * @return true if successful, false otherwise
     * TODO add when this method is called in Evolution.nextGen sequence
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

    /** Calculates the {@link #populationScore} of all members of this species, taking into account {@link #stag}
     * TODO add when this method is called in Evolution.nextGen sequence */
    protected void calculateScore() {
        populationScore = 0;
        for (Agent agent : population) {
            populationScore += agent.getScore();
        }
        populationScore /= population.size();
        if (stag >= Constants.maxStagDropoff) populationScore *= 0.7;
    }

    /**
     * Updates the {@link #stag} of this species.
     * <br>Stag increases by 1 every time the current generation does worse than the previous
     * <br>Stag decreases by a factor of how much better the current generation does than the previous
     * TODO add when this method is called in Evolution.nextGen sequence
     */
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
     * clears the {@link #population} arraylist
     * TODO add when this method is called in Evolution.nextGen sequence
     */
    protected void reset() {
        representative = population.get((int) (Math.random() * population.size()));
        population.clear();
    }

    /**
     * Effect: A percentage of the worst performing members of this species have their genome removed
     * TODO add when this method is called in Evolution.nextGen sequence
     */
    protected void cull() {
        int numSurvived = (int) (Math.round(population.size() * (1 - Constants.perctCull)));
        for (int i = population.size() - 1; i > numSurvived; i--) {
            population.remove(i).removeGenome();
        }
    }

    /**
     * Returns if this species has no current members
     */
    protected boolean isEmpty() {
        return population.isEmpty();
    }

    /** TODO
     * TODO add when this method is called in Evolution.nextGen sequence */
    protected void populateGenome(Agent emptyAgent) {
        Agent first = WeightedRandom.getRandom(population);
        Agent second = WeightedRandom.getRandom(population);
        first.crossover(second, emptyAgent);
        emptyAgent.reset();
        population.add(emptyAgent);
    }

    /** TODO
     * TODO add when this method is called in Evolution.nextGen sequence */
    @Override
    public double getScore() {
        return populationScore;
    }
}
