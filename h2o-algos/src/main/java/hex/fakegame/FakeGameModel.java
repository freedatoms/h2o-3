package hex.fakegame;

import hex.Model;
import hex.ModelCategory;
import hex.ModelMetrics;
import water.H2O;
import water.Key;

public class FakeGameModel extends Model<FakeGameModel,FakeGameModel.FakeGameParameters,FakeGameModel.FakeGameOutput> {

  public static class FakeGameParameters extends Parameters {
    public String algoName() { return "FakeGame"; }
    public String fullName() { return "FakeGame"; }
    public String javaName() { return FakeGameModel.class.getName(); }
    @Override public long progressUnits() { return _max_iterations; }
    public int _max_iterations = 1000; // Max iterations
  }

  public static class FakeGameOutput extends Output {
    // Iterations executed
    public int _iterations;
    public double[] _maxs;
    public FakeGameOutput( FakeGame b ) { super(b); }
    @Override public ModelCategory getModelCategory() { return ModelCategory.Unknown; }
  }

  FakeGameModel( Key selfKey, FakeGameParameters parms, FakeGameOutput output) { super(selfKey,parms,output); }

  @Override
  public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    throw H2O.unimpl("No Model Metrics for FakeGameModel.");
  }

  @Override protected double[] score0(double data[/*ncols*/], double preds[/*nclasses+1*/]) {
    throw H2O.unimpl();
  }

}
