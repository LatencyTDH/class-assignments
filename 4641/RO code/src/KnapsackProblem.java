import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class KnapsackProblem {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;

    private static final int ITERATIONS = 100;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws Exception{
        String filename = String.format("%s-%d.txt", "KnapsackProblem", System.currentTimeMillis());
        File file = new File(filename);
        FileOutputStream fo = new FileOutputStream(file);
        PrintStream ps = new PrintStream(fo);
        System.setOut(ps);

        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
         int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        long start;
        long rhcTime = 0, saTime = 0, GATime = 0, MIMICTime = 0, temp;
        int rhcCount = 0, saCount = 0, GACount = 0, MIMICCount = 0; //tracks which RO alg has the best fitness for each iteration
        double rhcOPT = 0, saOPT = 0, GAOPT = 0, MIMICOPT = 0;

        for (int i = 0; i < ITERATIONS; i++) {
        	System.out.println(i);
	        start = System.currentTimeMillis();
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
	        fit.train();
            rhcOPT = ef.value(rhc.getOptimal());
            System.out.println("RHC: " + rhcOPT);
            temp = System.currentTimeMillis() - start;
            rhcTime += temp;
            System.out.println("Runtime: " + temp);
	        
	        start = System.currentTimeMillis();
	        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
	        fit = new FixedIterationTrainer(sa, 200000);
	        fit.train();
            saOPT = ef.value(sa.getOptimal());
            System.out.println("SA: " + saOPT);
            temp = System.currentTimeMillis() - start;
            saTime += temp;
            System.out.println("Runtime: " + temp);
	        
	        start = System.currentTimeMillis();
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
	        fit = new FixedIterationTrainer(ga, 1000);
	        fit.train();
            GAOPT = ef.value(ga.getOptimal());
            System.out.println("GA: " + GAOPT);
            temp = System.currentTimeMillis() - start;
            GATime += temp;
            System.out.println("Runtime: " + temp);
	        
	        start = System.currentTimeMillis();
	        MIMIC mimic = new MIMIC(200, 100, pop);
	        fit = new FixedIterationTrainer(mimic, 1000);
	        fit.train();
            MIMICOPT = ef.value(mimic.getOptimal());
            System.out.println("MIMIC: " + MIMICOPT);
            temp = System.currentTimeMillis() - start;
            MIMICTime += temp;
            System.out.println("Runtime: " + temp);
	        System.out.println("\n\n");

            double max = Math.max(Math.max(Math.max(rhcOPT,saOPT),GAOPT),MIMICOPT);
            if (rhcOPT == max) rhcCount++;
            else if (saOPT == max) saCount++;
            else if (GAOPT == max) GACount++;
            else if (MIMICOPT == max) MIMICCount++;
        }
        System.out.println("Average time for all " + ITERATIONS + " iterations.\n");
        System.out.println("RHC: " + rhcTime/(ITERATIONS+0.0));
        System.out.println("SA: " + saTime/(ITERATIONS+0.0));
        System.out.println("GA: " + GATime/(ITERATIONS+0.0));
        System.out.println("MIMIC: " + MIMICTime/(ITERATIONS+0.0));
        System.out.println("\n");
        System.out.println(String.format("Best fitness over %d iterations:\n" +
                "RHC: %d\nSA: %d\nGA: %d\nMIMIC: %d"
                , ITERATIONS, rhcCount, saCount, GACount, MIMICCount));
    }

}
