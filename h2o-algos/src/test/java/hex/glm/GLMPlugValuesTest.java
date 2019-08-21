package hex.glm;

import org.junit.BeforeClass;
import org.junit.Test;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import water.fvec.TestFrameBuilder;

import static org.junit.Assert.*;

public class GLMPlugValuesTest extends TestUtil {

  @BeforeClass
  public static void setup() {
    stall_till_cloudsize(1);
  }

  @Test
  public void testBasic() {
    Scope.enter();
    try {
      // has missing
      Frame fr = new TestFrameBuilder()
              .withColNames("x", "y", "z")
              .withDataForCol(0, ard(1.0d, Double.NaN))
              .withDataForCol(1, ard(Double.NaN, 2.0d))
              .withDataForCol(2, ard(2.0d, 8.0d))
              .build();
      // missing values manually substituted with the corresponding plug values
      Frame fr2 = new TestFrameBuilder()
              .withColNames("x", "y", "z")
              .withDataForCol(0, ard(1.0d, 4.0d))
              .withDataForCol(1, ard(0.5d, 2.0d))
              .withDataForCol(2, ard(2.0d, 8.0d))
              .build();
      
      Frame plugValues = oneRowFrame(new String[]{"x", "y"}, new double[]{4.0d, 0.5d});

      GLMModel.GLMParameters params = new GLMModel.GLMParameters();
      params._response_column = "z";
      params._family = GLMModel.GLMParameters.Family.gaussian;
      params._standardize = false;
      params._train = fr._key;
      params._ignore_const_cols = false;
      params._intercept = false;
      params._seed = 42;
      
      GLMModel.GLMParameters params2 = (GLMModel.GLMParameters) params.clone();
      params2._train = fr2._key;
      
      params._missing_values_handling = GLMModel.GLMParameters.MissingValuesHandling.PlugValues;
      params._plug_values = plugValues._key;
      GLMModel model = new GLM(params).trainModel().get();
      Scope.track_generic(model);

      GLMModel model2 = new GLM(params2).trainModel().get();
      Scope.track_generic(model2);

      assertEquals(model2.coefficients(), model.coefficients());
    } finally {
      Scope.exit();
    }
  }
  
  @Test
  public void testPlugValues_zeros() {
    Scope.enter();
    try {
      Frame fr = parse_test_file("smalldata/junit/cars.csv");
      Scope.track(fr);
      fr.remove("name");

      // check that we actually do have some NAs in the dataset
      assertTrue(fr.vec("economy (mpg)").naCnt() > 0);
      
      GLMModel.GLMParameters params = new GLMModel.GLMParameters(
              GLMModel.GLMParameters.Family.poisson,
              GLMModel.GLMParameters.Family.poisson.defaultLink, 
              new double[]{0}, new double[]{0},0,0);
      params._response_column = "power (hp)";
      params._train = fr._key;
      params._lambda = new double[]{0};
      params._alpha = new double[]{0};
      params._missing_values_handling = GLMModel.GLMParameters.MissingValuesHandling.MeanImputation;
      params._seed = 42;

      GLMModel.GLMParameters params_means = (GLMModel.GLMParameters) params.clone();
      GLMModel.GLMParameters params_zeros = (GLMModel.GLMParameters) params.clone();

      GLMModel model = new GLM(params).trainModel().get();
      Scope.track_generic(model);

      Frame predictors = fr.clone();
      predictors.remove(params._response_column);
      Frame plugValues = oneRowFrame(predictors.names(), predictors.means());
      params_means._plug_values = plugValues._key;
      params_zeros._missing_values_handling = GLMModel.GLMParameters.MissingValuesHandling.PlugValues;

      // this doesn't check much - only demonstrates that setting means doesn't produce different results
      GLMModel model_means = new GLM(params_means).trainModel().get();
      Scope.track_generic(model_means);

      assertArrayEquals(model.beta(), model_means.beta(), 0);
      assertArrayEquals(model.dinfo()._numNAFill, model_means.dinfo()._numNAFill, 0);

      Frame plugValues_zeros = oneRowFrame(predictors.names(), new double[predictors.numCols()]);
      params_zeros._plug_values = plugValues_zeros._key;
      params_zeros._missing_values_handling = GLMModel.GLMParameters.MissingValuesHandling.PlugValues;

      GLMModel model_zeros = new GLM(params_zeros).trainModel().get();
      Scope.track_generic(model_zeros);
      
      // NA fill should be properly populated
      assertArrayEquals(model_zeros.dinfo()._numNAFill, new double[predictors.numCols()], 0);
      assertNotEquals(model.coefficients().get("economy (mpg)"), model_zeros.coefficients().get("economy (mpg)"));
    } finally {
      Scope.exit();
    }
  }

  private static Frame oneRowFrame(String[] names, double[] values) {
    TestFrameBuilder builder = new TestFrameBuilder().withColNames(names);
    for (int i = 0; i < values.length; i++)
      builder.withDataForCol(i, new double[]{values[i]});
    return builder.build();
  }
  
}
