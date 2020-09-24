import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.oracle.labs.mlrg.olcut.config.json.JsonProvenanceModule;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

public class ClassificationExample {
    public static void main(String[] args) throws IOException {

        LabelFactory labelFactory = new LabelFactory();
        CSVLoader csvLoader = new CSVLoader<>(labelFactory);

        String[] irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
        ListDataSource irisesSource = csvLoader.loadDataSource(Paths.get("bezdekIris.data"), "species", irisHeaders);
        TrainTestSplitter irisSplitter = new TrainTestSplitter<>(irisesSource, 0.7, 1L);

        MutableDataset trainingDataset = new MutableDataset<>(irisSplitter.getTrain());
        MutableDataset testingDataset = new MutableDataset<>(irisSplitter.getTest());
        System.out.println(String.format("Training data size = %d, number of features = %d, number of classes = %d",
                trainingDataset.size(), trainingDataset.getFeatureMap().size(), trainingDataset.getOutputInfo().size()));
        System.out.println(String.format("Testing data size = %d, number of features = %d, number of classes = %d",
                testingDataset.size(), testingDataset.getFeatureMap().size(), testingDataset.getOutputInfo().size()));

        Trainer<Label> trainer = new LogisticRegressionTrainer();
        //Trainer<Label> trainer = new XGBoostClassificationTrainer(2);
        System.out.println(trainer);

        Model<Label> irisModel = trainer.train(trainingDataset);

        LabelEvaluator evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(irisModel, testingDataset);
        System.out.println(evaluation);
        System.out.println(evaluation.getConfusionMatrix());

        List<Example> data = testingDataset.getData();
        for (Example<Label> testingData : data) {
            Prediction<Label> predict = irisModel.predict(testingData);
            String expectedResult = testingData.getOutput().getLabel();
            String predictResult = predict.getOutput().getLabel();
            if (!predictResult.equals(expectedResult)) {
                System.out.println("Expected result : " + expectedResult);
                System.out.println("Predicted result: " + predictResult);
                System.out.println(predict.getOutputScores());
            }
        }

        ImmutableFeatureMap featureMap = irisModel.getFeatureIDMap();
        for (VariableInfo v : featureMap) {
            System.out.println(v);
        }

        ModelProvenance provenance = irisModel.getProvenance();
        System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getDatasetProvenance().getSourceProvenance()));
        System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getTrainerProvenance()));

        ObjectMapper objMapper = new ObjectMapper();
        objMapper.registerModule(new JsonProvenanceModule());
        objMapper = objMapper.enable(SerializationFeature.INDENT_OUTPUT);
        String jsonProvenance = objMapper.writeValueAsString(ProvenanceUtil.marshalProvenance(provenance));
        System.out.println(jsonProvenance);
        System.out.println(irisModel);

        String jsonEvaluationProvenance = objMapper.writeValueAsString(ProvenanceUtil.convertToMap(evaluation.getProvenance()));
        System.out.println(jsonEvaluationProvenance);
    }
}
