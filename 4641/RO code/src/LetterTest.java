import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying a letter to be "O" or not "O".
 *
 * @author modified by Sean
 * @version 1.0
 */
public class LetterTest {
    private Instance[] instances = initializeInstances();
    private final String DATASET = "letter_entire_new.txt";

    private int inputLayer , hiddenLayer , outputLayer , trainingIterations ;
    private BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private ErrorMeasure measure = new SumOfSquaresError();

    private DataSet set = new DataSet(instances);

    private BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private String[] oaNames = {"RHC", "SA", "GA"};
    private String results = "";

    private DecimalFormat df = new DecimalFormat("0.000");
    
    public LetterTest(int inputLayer, int hiddenLayer, int outputLayer, int trainingIterations) {
        this.inputLayer = inputLayer;
        this.hiddenLayer = hiddenLayer;
        this.outputLayer = outputLayer;
        this.trainingIterations = trainingIterations;
    }
    public void run(String[] args) throws FileNotFoundException{
        /* Logs the data from the run */
        String filename = String.format("results-iterations-%d-%d.txt", trainingIterations, System.currentTimeMillis());
        File file = new File("result_data/" + filename);
        FileOutputStream fo = new FileOutputStream(file);
        PrintStream ps = new PrintStream(fo);
        System.setOut(ps);
        
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
        System.out.println(String.format("inputLayer=%d, hiddenLayer=%d, outputLayer=%d, trainingIterations=%d", 
                inputLayer, hiddenLayer, outputLayer, trainingIterations));
        results = "";
    }

    private void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private Instance[] initializeInstances() {
        final int INSTANCES = 20000;
        final int ATTRIBUTES = 16;
        double[][][] attributes = new double[INSTANCES][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("data/"+DATASET)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[ATTRIBUTES]; // 16 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < ATTRIBUTES; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications are 1 if the letter is 'O' and 0 otherwise
            instances[i].setLabel(new Instance(attributes[i][1][0] == 0 ? 0 : 1));
        }
        
        return instances;
    }
}
