package Genome;

import java.util.function.Consumer;

/** Activation Function enum used to store evaluators functions */
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

    /** Evaluates the array of inputs using this Activation Function enum */
    public double[] evaluate(double[] input){
        if(this.equals(softMax)){
            evaluator.accept(input);
        }else{
            for(int i=0;i<input.length;i++){
                input[i] = hidden.valueOf(toString()).evaluate(input[i]);
            }
        }
        return input;
    }
}
