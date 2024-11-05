package Genome;

import java.util.function.Consumer;
import java.util.function.Function;

public class ActivationFunction {
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

        public double evaluate(double input){
            return evaluator.apply(input);
        }
    }
    public enum output{
        none(null),
        relu(null),
        sigmoid(null),
        tanh(null),
        leakyRelu(null),
        softMax((input)->{
            double sum=0;
            for (double num : input) {
                sum += num;
            }
            for(int i=0;i<input.length;i++){
                input[i]= sum==0 ? 0 : input[i]/sum;
            }
        });

        private final Consumer<double[]> evaluator;
        output(Consumer<double[]> evaluate){
            this.evaluator = evaluate;
        }

        public double[] evaluate(double[] input){
            if(this.equals(softMax)){
                evaluator.accept(input);
            }else{
                Function<Double,Double> evaluator = hidden.valueOf(toString()).evaluator;
                for(int i=0;i<input.length;i++){
                    input[i] = evaluator.apply(input[i]);
                }
            }
            return input;
        }
    }
}
