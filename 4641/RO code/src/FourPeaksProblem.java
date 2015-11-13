import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
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
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class FourPeaksProblem {
    private static final int N = 80;
    
    private static final int T = N/10;
    
    private static final int ITERATIONS = 100;
    
    public static void main(String[] args) throws Exception {
//        String filename = "results-" + System.currentTimeMillis() + ".txt";
        String filename = String.format("%s-%d.txt", "FourPeaksProblem", System.currentTimeMillis());
        File file = new File(filename);
        FileOutputStream fo = new FileOutputStream(file);
        PrintStream ps = new PrintStream(fo);
        System.setOut(ps);


        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
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
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
	        fit = new FixedIterationTrainer(ga, 1000);
	        fit.train();
            GAOPT = ef.value(ga.getOptimal());
            System.out.println("GA: " + GAOPT);
            temp = System.currentTimeMillis() - start;
            GATime += temp;
            System.out.println("Runtime: " + temp);
	        
	        start = System.currentTimeMillis();
	        MIMIC mimic = new MIMIC(200, 5, pop);
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
