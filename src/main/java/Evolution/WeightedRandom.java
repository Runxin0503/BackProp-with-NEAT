package Evolution;

import java.util.List;


/**
 * A functional interface representing an object that can be assigned a score,
 * allowing it to participate in weighted random selection.
 * <p>
 * Classes implementing this interface can be selected proportionally to their score using {@link #getRandom(List)}.
 * </p>
 */
@FunctionalInterface
public interface WeightedRandom {
    /**
     * Returns the score associated with this object.
     * <p>
     * Higher scores increase the probability of being selected during weighted random sampling.
     * </p>
     *
     * @return the score value
     */
    double getScore();

    /**
     * Selects an element from a list based on weighted random sampling.
     * <p>
     * Elements with higher {@code getScore()} values are more likely to be selected.
     * If all scores are zero, an element is chosen uniformly at random.
     * </p>
     *
     * @param list the list of candidates to sample from
     * @param <T> a type that implements {@link WeightedRandom}
     * @return a randomly selected element based on the weights
     * @throws IllegalArgumentException if the list is empty
     * @throws RuntimeException if an unexpected error occurs during sampling
     */
    static <T extends WeightedRandom> T getRandom(List<T> list) {
        if (list.isEmpty()) throw new IllegalArgumentException("List is empty");
        double totalValue = 0;
        for (T weightedRandom : list) totalValue += weightedRandom.getScore();
        if (totalValue == 0) return list.get((int) (Math.random() * list.size()));

        double randomValue = totalValue * Math.random();
        for (T weightedRandom : list) if ((randomValue -= weightedRandom.getScore()) < 0) return weightedRandom;

        throw new RuntimeException("Unexpected occurrence");
    }
}
