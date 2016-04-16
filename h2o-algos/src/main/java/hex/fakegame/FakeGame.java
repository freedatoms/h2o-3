package hex.fakegame;

import configuration.classifiers.ClassifierConfig;
import configuration.classifiers.single.ClassifierModelConfig;
import configuration.game.trainers.QuasiNewtonConfig;
import configuration.models.ensemble.BaseModelsDefinition;
import configuration.models.single.PolynomialModelConfig;
import game.classifiers.Classifier;
import game.classifiers.ClassifierFactory;
import game.classifiers.ConnectableClassifier;
import game.data.AbstractGameData;
import game.data.ArrayGameData;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.fakegame.FakeGameModel.FakeGameOutput;
import hex.fakegame.FakeGameModel.FakeGameParameters;
import water.MRTask;
import water.Scope;
import water.fvec.Chunk;
import water.util.Log;


import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;


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


  public enum ClassifierType {
    ClassifierArbitrating,
    ClassifierBagging,
    ClassifierBoosting,
    ClassifierCascadeGen,
    ClassifierCascadeGenProb,
    ClassifierCascading,
    ClassifierDelegating,
    ClassifierEnsembleBase,
    ClassifierEnsemble,
    ClassifierEvolvableEnsemble,
    ClassifierGAME,
    ClassifierStacking,
    ClassifierStackingProb,
    ClassifierWeighted
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
    public void compute2() {
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
        LinkedList<Classifier> cls = (new FakeGameLearner(resp_col, isClassifier())).doAll(_parms.train())._lfg;

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
    // OUT
    LinkedList<Classifier> _lfg;

    FakeGameLearner(int resp_col, boolean isClassifier) {
      this.resp_col = resp_col;
      this.isClassifier = isClassifier;
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
        ClassifierModelConfig clc = new ClassifierModelConfig();
        clc.setClassModelsDef(BaseModelsDefinition.UNIFORM);

        PolynomialModelConfig smci;
        smci = new PolynomialModelConfig();
        smci.setTrainerClassName("QuasiNewtonTrainer");
        smci.setTrainerCfg(new QuasiNewtonConfig());
        smci.setMaxDegree(5);
        clc.addClassModelCfg(smci);

        clc.setDescription("Polynomial classifier with max degree 5");

        int outputs = data.getONumber();
        clc.setModelsNumber(outputs);

        Classifier c = ClassifierFactory.createNewClassifier(clc, data, true);

/*
        System.out.println(((ConnectableClassifier) c).toEquation());
        DecimalFormat formater = new DecimalFormat("#.#");
        double err = 0;
        for (int j = 0; j < data.getInstanceNumber(); j++) {
          data.publishVector(j);
          double[] out = ((ConnectableClassifier) c).getOutputProbabilities();
          /*String outs = "{";
          for (int i = 0; i < data.getONumber(); i++) {
            String o = formater.format(out[i]);
            String d = formater.format(data.getTargetOutput(i));
            outs += "[" + o + "|" + d + "]";
          }
          outs += ")";
          System.out.println("props[out|target]:" + outs + "class predicted:" + ((ConnectableClassifier) c).getOutput());
          /*/
        /*
          for (int i = 0; i < data.getONumber(); i++) {
            err += Math.pow(out[i] - data.getTargetOutput(i), 2);

          }
        }
        System.out.println("Overall RMS Error: " + Math.sqrt(err / data.getInstanceNumber()));
*/
        _lfg.add(c);
      }
    }

    @Override
    public void reduce(FakeGameLearner mrt) {
      if (mrt._lfg.size() >0 && this._lfg != mrt._lfg)
        Log.debug("FakeGame reducing "+_lfg.size()+" + "+mrt._lfg.size());

      _lfg.addAll(mrt._lfg);
    }
  }
}
