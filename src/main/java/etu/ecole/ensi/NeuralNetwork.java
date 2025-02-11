package etu.ecole.ensi;

import java.util.Arrays;
import java.util.Random;
/**
 * This class represent a Neural Network based on sigmoid activation method
 * @author Mohamed AKADDAR
 * @Date 11/02/2025
 *
 * PS : Please accept misunderstand or uncomplete work, it's about first time with NN
 * */
public class NeuralNetwork {
    private final int nbHidden;
    private final int nbOutputs;

    private final double[][] weightsIH; // Poids entre couche d'entrée et couche cachée
    private final double[][] weightsHO; // Poids entre couche cachée et couche de sortie

    private final double learningRate;
    private final Random generator = new Random();

    /**
     * Constructeur du réseau de neurones
     * @param nbInputs Nombre de neurones en entrée
     * @param nbHidden Nombre de neurones cachés
     * @param nbOutputs Nombre de neurones en sortie
     * @param learningRate Taux d'apprentissage
     */
    public NeuralNetwork(int nbInputs, int nbHidden, int nbOutputs, double learningRate) {
        this.nbHidden = nbHidden;
        this.nbOutputs = nbOutputs;
        this.learningRate = learningRate;

        // Initialisation des poids avec des valeurs aléatoires
        weightsIH = initWeights(nbHidden, nbInputs);
        weightsHO = initWeights(nbOutputs, nbHidden);
    }

    /**
     * Initialise une matrice de poids avec des valeurs aléatoires
     */
    private double[][] initWeights(int rows, int cols) {
        double[][] weights = new double[rows][cols];
        for (double[] row : weights) {
            Arrays.setAll(row, i -> generator.nextGaussian()); // Distribution normale
        }
        return weights;
    }

    /**
     * Fonction d'activation Sigmoïde
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Dérivée de la fonction Sigmoïde
     */
    private double dsigmoid(double y) {
        return y * (1 - y);
    }

    /**
     * Propagation avant : Calcul de la sortie à partir d'une entrée
     */
    public double[] feedForward(double[] input) {
        double[] hidden = activateLayer(input, weightsIH);
        return activateLayer(hidden, weightsHO);
    }

    /**
     * Active une couche neuronale
     */
    private double[] activateLayer(double[] inputs, double[][] weights) {
        return Arrays.stream(weights)
                .mapToDouble(neuronWeights -> sigmoid(dotProduct(neuronWeights, inputs)))
                .toArray();
    }

    /**
     * Produit scalaire entre un vecteur de poids et un vecteur d'entrée
     */
    private double dotProduct(double[] weights, double[] inputs) {
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    /**
     * Entraînement du réseau de neurones avec rétropropagation de l'erreur
     */
    public double train(double[] input, double[] target) {
        // Propagation avant
        double[] hidden = activateLayer(input, weightsIH);
        double[] outputs = activateLayer(hidden, weightsHO);

        // Calcul des erreurs de sortie
        double[] outputErrors = computeErrors(target, outputs);

        // Mise à jour des poids sortie → cachée
        adjustWeights(weightsHO, outputErrors, outputs, hidden);

        // Calcul des erreurs de la couche cachée
        double[] hiddenErrors = computeHiddenErrors(weightsHO, outputErrors);

        // Mise à jour des poids entrée → cachée
        adjustWeights(weightsIH, hiddenErrors, hidden, input);

        // Retourne l'erreur quadratique moyenne (MSE)
        return meanSquaredError(outputErrors);
    }

    /**
     * Calcul des erreurs pour une couche
     */
    private double[] computeErrors(double[] target, double[] outputs) {
        double[] errors = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            errors[i] = target[i] - outputs[i];
        }
        return errors;
    }

    /**
     * Mise à jour des poids
     */
    private void adjustWeights(double[][] weights, double[] errors, double[] layerOutputs, double[] prevLayerOutputs) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] += learningRate * errors[i] * dsigmoid(layerOutputs[i]) * prevLayerOutputs[j];
            }
        }
    }

    /**
     * Calcul des erreurs pour la couche cachée
     */
    private double[] computeHiddenErrors(double[][] weightsHO, double[] outputErrors) {
        double[] hiddenErrors = new double[nbHidden];
        for (int i = 0; i < nbHidden; i++) {
            double error = 0;
            for (int j = 0; j < nbOutputs; j++) {
                error += weightsHO[j][i] * outputErrors[j];
            }
            hiddenErrors[i] = error;
        }
        return hiddenErrors;
    }

    /**
     * Calcule l'erreur quadratique moyenne (MSE)
     */
    private double meanSquaredError(double[] errors) {
        double sum = 0;
        for (double error : errors) {
            sum += error * error;
        }
        return sum / errors.length;
    }

    /**
     * Programme principal
     */
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 6, 1, 0.1);

        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{0}, {1}, {1}, {0}};

        int epochs = 1000000;  // Nombre d'itérations d'entraînement
        double[] errors = new double[epochs];

        for (int i = 0; i < epochs; i++) {
            int index = new Random().nextInt(inputs.length);
            errors[i] = nn.train(inputs[index], targets[index]);
        }

        // Affichage des résultats après l'entraînement
        for (double[] input : inputs) {
            double[] output = nn.feedForward(input);
            System.out.printf("%.1f XOR %.1f -> %.4f%n", input[0], input[1], output[0]);
        }

        // Affichage de l'évolution de l'erreur d'entraînement
        plotLearningCurve(errors);
    }

    /**
     * Trace la courbe d'apprentissage en sauvegardant les erreurs
     */
    private static void plotLearningCurve(double[] errors) {
        try {
            java.nio.file.Files.write(
                    java.nio.file.Paths.get("learning_curve.csv"),
                    Arrays.toString(errors).replaceAll("[\\[\\] ]", "").replace(",", "\n").getBytes()
            );
            System.out.println("Courbe d'apprentissage sauvegardée dans 'learning_curve.csv'.");
        } catch (Exception e) {
            System.out.println("Erreur lors de la sauvegarde des données : " + e.getMessage());
        }
    }
}
