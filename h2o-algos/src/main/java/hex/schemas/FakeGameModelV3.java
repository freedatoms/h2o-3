package hex.schemas;

import hex.fakegame.FakeGameModel;
import water.api.API;
import water.api.ModelOutputSchema;
import water.api.ModelSchema;

public class FakeGameModelV3 extends ModelSchema<FakeGameModel, FakeGameModelV3, FakeGameModel.FakeGameParameters, FakeGameV3.FakeGameParametersV3, FakeGameModel.FakeGameOutput, FakeGameModelV3.FakeGameModelOutputV3> {

  public static final class FakeGameModelOutputV3 extends ModelOutputSchema<FakeGameModel.FakeGameOutput, FakeGameModelOutputV3> {
    // Output fields

  } // FakeGameModelOutputV2


  //==========================
  // Custom adapters go here

  // TOOD: I think we can implement the following two in ModelSchema, using reflection on the type parameters.
  public FakeGameV3.FakeGameParametersV3 createParametersSchema() { return new FakeGameV3.FakeGameParametersV3(); }
  public FakeGameModelOutputV3 createOutputSchema() { return new FakeGameModelOutputV3(); }
}
