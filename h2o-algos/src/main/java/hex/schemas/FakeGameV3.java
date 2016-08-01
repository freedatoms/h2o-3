package hex.schemas;

import hex.fakegame.FakeGame;
import hex.fakegame.FakeGameModel;
import water.api.API;
import water.api.ModelParametersSchema;

import java.io.File;
import java.nio.file.Path;

public class FakeGameV3 extends ModelBuilderSchema<FakeGame, FakeGameV3, FakeGameV3.FakeGameParametersV3> {

  public static final class FakeGameParametersV3 extends ModelParametersSchema<FakeGameModel.FakeGameParameters, FakeGameParametersV3> {
    static public String[] fields = new String[]{
        "training_frame",
        "response_column",
        "classifier_config",
        "ignored_columns"
    };

    @API(help = "Classifier config")
  public String classifier_config;

  } // FakeGameParametersV2

}
