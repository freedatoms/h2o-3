package hex.fakegame;

import game.classifiers.Classifier;
import hex.*;
import water.H2O;
import water.Key;

import java.util.LinkedList;

public class FakeGameModel extends Model<FakeGameModel, FakeGameModel.FakeGameParameters, FakeGameModel.FakeGameOutput> {

    public static class FakeGameParameters extends Model.Parameters {
        @Override
        public String algoName() {
            return "FakeGame";
        }

        @Override
        public String fullName() {
            return "FakeGame";
        }

        @Override
        public String javaName() {
            return FakeGameModel.class.getName();
        }

        @Override
        public long progressUnits() {
            return 1;
        }

        public String _classifier_config;
    }

    public static class FakeGameOutput extends Model.Output {

        public LinkedList<Classifier> _cls;

        public FakeGameOutput(FakeGame b) {
            super(b);
        }

        @Override
        public ModelCategory getModelCategory() {
            if (isClassifier()) {
                return (nclasses() > 2) ? ModelCategory.Multinomial : ModelCategory.Binomial;
            }
            return ModelCategory.Regression;
        }

        @Override
        public boolean isSupervised() {
            return true;
        }
    }

    FakeGameModel(Key selfKey, FakeGameParameters parms, FakeGameOutput output) {
        super(selfKey, parms, output);
    }

    @Override
    public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
        switch (_output.getModelCategory()) {
            case Binomial:
                return new ModelMetricsBinomial.MetricBuilderBinomial(domain);
            case Multinomial:
                return new ModelMetricsMultinomial.MetricBuilderMultinomial(domain.length, domain);
            default:
                throw H2O.unimpl();
        }
    }

    @Override
    protected double[] score0(double data[/*ncols*/], double preds[/*nclasses+1*/]) {
        for (int i = 0; i < _output.nclasses() + 1; i++) {
            preds[i] = 0.0;
        }

        for (Classifier c : _output._cls) {
            preds[1 + c.getOutput(data)] += 1.0 / _output._cls.size();
        }

        double max = -1.0;
        for (int i = 1; i < _output.nclasses() + 1; i++) {
            if (preds[i] > max)
                max = preds[i];
        }

        for (int i = 1; i < _output.nclasses() + 1; i++) {
            if (preds[i] == max) {
                preds[0] = i;
                break;
            }
        }
        return preds;
    }

}
