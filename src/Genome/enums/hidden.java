package Genome.enums;

import java.util.function.Function;

/** Activation Function enum used to store evaluators functions */
public enum hidden{
        none((input)->input),
        relu((input)->(input > 0) ? input : 0),
        sigmoid((input)->1/(1+Math.pow(Math.E,-input))),
        tanh((input)->(Math.pow(Math.E,input)-Math.pow(Math.E,-input))/(Math.pow(Math.E,input)+Math.pow(Math.E,-input))),
        leakyRelu((input)-> Math.max(input, 0.1 * input));

        private final Function<Double,Double> evaluator;
        hidden(Function<Double,Double> evaluate){
            this.evaluator = evaluate;
        }

        /** Evaluates this input using this Activation Function enum */
        public double evaluate(double input){
            return evaluator.apply(input);
        }
}

