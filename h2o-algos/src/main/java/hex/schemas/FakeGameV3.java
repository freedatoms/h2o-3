package hex.schemas;

import hex.fakegame.FakeGame;
import hex.fakegame.FakeGameModel;
import water.api.API;
import water.api.ModelParametersSchema;

public class FakeGameV3 extends ModelBuilderSchema<FakeGame, FakeGameV3, FakeGameV3.FakeGameParametersV3> {

  public static final class FakeGameParametersV3 extends ModelParametersSchema<FakeGameModel.FakeGameParameters, FakeGameParametersV3> {
    static public String[] fields = new String[]{"training_frame", "response_column", "classifier_type", "ignored_columns"};

    // Input fields
    //@API(help="Maximum training iterations.")  public int max_iterations;

    @API(help = "Classifier type", values = {"ClassifierArbitrating",
        "ClassifierBagging",
        "ClassifierBoosting",
        "ClassifierCascadeGen",
        "ClassifierCascadeGenProb",
        "ClassifierCascading",
        "ClassifierDelegating",
        "ClassifierEnsembleBase",
        "ClassifierEnsemble",
        "ClassifierEvolvableEnsemble",
        "ClassifierGAME",
        "ClassifierStacking",
        "ClassifierStackingProb",
        "ClassifierWeighted"})
  public FakeGame.ClassifierType classifier_type;

  } // FakeGameParametersV2

}
