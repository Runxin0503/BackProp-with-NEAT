package Evolution;

import java.util.List;

@FunctionalInterface
public interface WeightedRandom {
    double getValue();

    static <T extends WeightedRandom> T getRandom(List<T> list) {
        if(list.isEmpty()) throw new IllegalArgumentException("List is empty");
        double totalValue = 0;
        for (T weightedRandom : list) totalValue+=weightedRandom.getValue();
        if(totalValue==0) return list.get((int) (Math.random() * list.size()));

        double randomValue = totalValue * Math.random();
        for(T weightedRandom : list) if((randomValue-=weightedRandom.getValue()) < 0) return weightedRandom;

        throw new RuntimeException("Unexpected occurrence");
    }
}
