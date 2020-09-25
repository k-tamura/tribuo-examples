import org.tribuo.*;
import org.tribuo.math.la.DenseVector;
import org.tribuo.util.Util;
import org.tribuo.clustering.*;
import org.tribuo.clustering.evaluation.*;
import org.tribuo.clustering.example.ClusteringDataGenerator;
import org.tribuo.clustering.kmeans.*;
import org.tribuo.clustering.kmeans.KMeansTrainer.Distance;

public class ClusteringExample {
    public static void main(String[] args) {
        ClusteringEvaluator eval = new ClusteringEvaluator();
        Dataset<ClusterID> data = ClusteringDataGenerator.gaussianClusters(500, 1L);
        Dataset<ClusterID> test = ClusteringDataGenerator.gaussianClusters(500, 2L);
        KMeansTrainer trainer = new KMeansTrainer(5,10,Distance.EUCLIDEAN,1,1);
        long startTime = System.currentTimeMillis();
        KMeansModel model = trainer.train(data);
        long endTime = System.currentTimeMillis();
        System.out.println("Training with 5 clusters took " + Util.formatDuration(startTime,endTime));
        DenseVector[] centroids = model.getCentroidVectors();
        for (DenseVector centroid : centroids) {
            System.out.println(centroid);
        }
        ClusteringEvaluation trainEvaluation = eval.evaluate(model, data);
        System.out.println(trainEvaluation);
        ClusteringEvaluation testEvaluation = eval.evaluate(model, test);
        System.out.println(testEvaluation);
    }
}
