package hex.fakegame;

import configuration.CfgTemplate;
import configuration.ConfigurationFactory;
import game.classifiers.Classifier;
import game.classifiers.ClassifierFactory;
import game.data.ArrayGameData;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.fakegame.FakeGameModel.FakeGameOutput;
import hex.fakegame.FakeGameModel.FakeGameParameters;
import org.apache.commons.lang.NotImplementedException;
import water.MRTask;
import water.Scope;
import water.fvec.Chunk;
import water.util.Log;

import java.io.StringReader;
import java.util.LinkedList;


public class FakeGame extends ModelBuilder<FakeGameModel, FakeGameParameters, FakeGameOutput> {
  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{
        ModelCategory.Regression,
        ModelCategory.Binomial,
        ModelCategory.Multinomial
    };
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Stable;
  }



  // Called from Nano thread; start the FakeGame Job on a F/J thread
  public FakeGame(boolean startup_once) {
    super(new FakeGameParameters(), startup_once);
  }

  public FakeGame(FakeGameParameters parms) {
    super(parms);
    init(false);
  }

  @Override
  protected FakeGameDriver trainModelImpl() {
    return new FakeGameDriver();
  }

  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (!isClassifier()) {
      this.hide("_classifier_type", "Classifier type is not used when doing regression");
    }
  }

  // ----------------------
  private class FakeGameDriver extends Driver {
    @Override
    public void computeImpl() {
      FakeGameModel model = null;
      try {
        Scope.enter();
        _parms.read_lock_frames(_job); // Fetch & read-lock source frame
        init(true);

        // The model to be built
        model = new FakeGameModel(_job._result, _parms, new FakeGameOutput(FakeGame.this));
        model.delete_and_lock(_job);


        String[] names = _parms.train()._names;
        int resp_col = 0;
        for (int i = 0; i < names.length; i++) {
          if (_parms._response_column.equals(names[i])) {
            resp_col = i;
            break;
          }
        }
        LinkedList<Classifier> cls = (new FakeGameLearner(resp_col, isClassifier(), _parms._classifier_config)).doAll(_parms.train())._lfg;

        // Fill in the model
        model._output._cls = cls;
        model.update(_job);   // Update model in K/V store
        _job.update(1);       // One unit of work

      } finally {
        if (model != null) model.unlock(_job);
        _parms.read_unlock_frames(_job);
        Scope.exit(model == null ? null : model._key);
      }
      tryComplete();
    }
  }


  private static class FakeGameLearner extends MRTask<FakeGameLearner> {
    // IN
    int resp_col;
    boolean isClassifier;
    String classifier_config;
    // OUT
    LinkedList<Classifier> _lfg;

    FakeGameLearner(int resp_col, boolean isClassifier, String classifier_cfg) {
      this.resp_col = resp_col;
      this.isClassifier = isClassifier;
      this.classifier_config = classifier_cfg;
      _lfg = new LinkedList<>();
    }

    @Override
    public void map(Chunk[] cs) {
      double[][] inputVect = new double[cs[0]._len][cs.length - 1];
      double[][] target;
      if (isClassifier) {
        target = new double[cs[0]._len][cs[resp_col].vec().cardinality()];
      } else {
        target = new double[cs[0]._len][1];
      }

      for (int col = 0; col < resp_col; col++)
        for (int row = 0; row < cs[col]._len; row++)
          inputVect[row][col] = cs[col].atd(row);

      if (isClassifier) {
        for (int row = 0; row < cs[resp_col]._len; row++)
          target[row][(int) cs[resp_col].at8(row)] = 1;
      } else {
        for (int row = 0; row < cs[resp_col]._len; row++)
          target[row][0] = cs[resp_col].atd(row);
      }

      for (int col = resp_col + 1; col < cs.length; col++)
        for (int row = 0; row < cs[col]._len; row++)
          inputVect[row][col] = cs[col].atd(row);

      ArrayGameData data = new ArrayGameData(inputVect, target);


      if (isClassifier) {
        StringReader sr = new StringReader(classifier_config);
        CfgTemplate cfg = ConfigurationFactory.readConfiguration(sr);
        Classifier c = ClassifierFactory.createNewClassifier(cfg, data, true);
        _lfg.add(c);
      } else {
        throw new NotImplementedException("regression is not yet implemented");
      }
    }

    @Override
    public void reduce(FakeGameLearner mrt) {
      if (mrt._lfg.size() >0 && this._lfg != mrt._lfg) {
        Log.debug("FakeGame reducing " + _lfg.size() + " + " + mrt._lfg.size());
        _lfg.addAll(mrt._lfg);
      }
    }
  }
}
