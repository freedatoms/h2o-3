import sys
sys.path.insert(1,"../../")
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from tests import pyunit_utils

def pubdev_6396():
    user_specified_parameters = dict(ntrees=3, seed=1234)
    inferred_parameters = dict(distribution='gaussian')

    mnist_original = h2o.import_file(pyunit_utils.locate("smalldata/flow_examples/mnist/test.csv.gz"))
    predictors = mnist_original.columns[0:-1]
    target = 'C785'
    train, new_data = mnist_original.split_frame(ratios=[.5], seed=1234)
    drf = H2ORandomForestEstimator(model_id='drf', **user_specified_parameters)
    
    assert drf.non_default_parameters == dict()
    assert drf.user_specified_parameters == user_specified_parameters
    
    drf.train(x=predictors, y=target, training_frame=train)
    
    assert drf.non_default_parameters == dict(**user_specified_parameters, **inferred_parameters)    
    assert drf.user_specified_parameters == user_specified_parameters


if __name__ == "__main__":
    pyunit_utils.standalone_test(pubdev_6396)
else:
    pubdev_6396()
