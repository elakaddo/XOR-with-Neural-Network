package etu.ecole.ensi;

import java.text.MessageFormat;
import java.util.Random;
/**
 * Main class which is executed to train the NN and show its result after training period
 * */
public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1, 0.1);
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; //
        double[][] targets = {{0}, {1}, {1}, {0}}; // targets are the correct and expected results (supervised learning)

        for (int i = 0; i < 10000; i++) {
            int index = new Random().nextInt(inputs.length);
            nn.train(inputs[index], targets[index]);
        }

        for (int i = 0; i < inputs.length; i++) {
            double[] output = nn.feedForward(inputs[i]);
            System.out.println(MessageFormat.format("{0} XOR {1} -> {2}", inputs[i][0], inputs[i][1], output[0]));
        }
    }

}