#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# This file is auto-generated by h2o-3/h2o-bindings/bin/gen_python.py
# Copyright 2016 H2O.ai;  Apache License Version 2.0 (see LICENSE for details)
#
from __future__ import absolute_import, division, print_function, unicode_literals

from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric
import h2o


class H2OXGBoostEstimator(H2OEstimator):
    """
    XGBoost

    Builds a eXtreme Gradient Boosting model using the native XGBoost backend.
    """

    algo = "xgboost"

    def __init__(self, **kwargs):
        super(H2OXGBoostEstimator, self).__init__()
        self._parms = {}
        names_list = {"model_id", "training_frame", "validation_frame", "nfolds", "keep_cross_validation_models",
                      "keep_cross_validation_predictions", "keep_cross_validation_fold_assignment",
                      "score_each_iteration", "fold_assignment", "fold_column", "response_column", "ignored_columns",
                      "ignore_const_cols", "offset_column", "weights_column", "stopping_rounds", "stopping_metric",
                      "stopping_tolerance", "max_runtime_secs", "seed", "distribution", "tweedie_power",
                      "categorical_encoding", "quiet_mode", "export_checkpoints_dir", "ntrees", "max_depth", "min_rows",
                      "min_child_weight", "learn_rate", "eta", "sample_rate", "subsample", "col_sample_rate",
                      "colsample_bylevel", "col_sample_rate_per_tree", "colsample_bytree", "max_abs_leafnode_pred",
                      "max_delta_step", "monotone_constraints", "score_tree_interval", "min_split_improvement", "gamma",
                      "nthread", "max_bins", "max_leaves", "min_sum_hessian_in_leaf", "min_data_in_leaf", "sample_type",
                      "normalize_type", "rate_drop", "one_drop", "skip_drop", "tree_method", "grow_policy", "booster",
                      "reg_lambda", "reg_alpha", "dmatrix_type", "backend", "gpu_id"}
        if "Lambda" in kwargs: kwargs["lambda_"] = kwargs.pop("Lambda")
        for pname, pvalue in kwargs.items():
            if pname == 'model_id':
                self._id = pvalue
                self._parms["model_id"] = pvalue
            elif pname in names_list:
                # Using setattr(...) will invoke type-checking of the arguments
                setattr(self, pname, pvalue)
            else:
                raise H2OValueError("Unknown parameter %s = %r" % (pname, pvalue))

    @property
    def training_frame(self):
        """
        Id of the training data frame.

        Type: ``H2OFrame``.
        """
        return self._parms.get("training_frame")

    @training_frame.setter
    def training_frame(self, training_frame):
        self._parms["training_frame"] = H2OFrame._validate(training_frame, 'training_frame')


    @property
    def validation_frame(self):
        """
        Id of the validation data frame.

        Type: ``H2OFrame``.
        """
        return self._parms.get("validation_frame")

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        self._parms["validation_frame"] = H2OFrame._validate(validation_frame, 'validation_frame')


    @property
    def nfolds(self):
        """
        Number of folds for K-fold cross-validation (0 to disable or >= 2).

        Type: ``int``  (default: ``0``).
        """
        return self._parms.get("nfolds")

    @nfolds.setter
    def nfolds(self, nfolds):
        assert_is_type(nfolds, None, int)
        self._parms["nfolds"] = nfolds


    @property
    def keep_cross_validation_models(self):
        """
        Whether to keep the cross-validation models.

        Type: ``bool``  (default: ``True``).
        """
        return self._parms.get("keep_cross_validation_models")

    @keep_cross_validation_models.setter
    def keep_cross_validation_models(self, keep_cross_validation_models):
        assert_is_type(keep_cross_validation_models, None, bool)
        self._parms["keep_cross_validation_models"] = keep_cross_validation_models


    @property
    def keep_cross_validation_predictions(self):
        """
        Whether to keep the predictions of the cross-validation models.

        Type: ``bool``  (default: ``False``).
        """
        return self._parms.get("keep_cross_validation_predictions")

    @keep_cross_validation_predictions.setter
    def keep_cross_validation_predictions(self, keep_cross_validation_predictions):
        assert_is_type(keep_cross_validation_predictions, None, bool)
        self._parms["keep_cross_validation_predictions"] = keep_cross_validation_predictions


    @property
    def keep_cross_validation_fold_assignment(self):
        """
        Whether to keep the cross-validation fold assignment.

        Type: ``bool``  (default: ``False``).
        """
        return self._parms.get("keep_cross_validation_fold_assignment")

    @keep_cross_validation_fold_assignment.setter
    def keep_cross_validation_fold_assignment(self, keep_cross_validation_fold_assignment):
        assert_is_type(keep_cross_validation_fold_assignment, None, bool)
        self._parms["keep_cross_validation_fold_assignment"] = keep_cross_validation_fold_assignment


    @property
    def score_each_iteration(self):
        """
        Whether to score during each iteration of model training.

        Type: ``bool``  (default: ``False``).
        """
        return self._parms.get("score_each_iteration")

    @score_each_iteration.setter
    def score_each_iteration(self, score_each_iteration):
        assert_is_type(score_each_iteration, None, bool)
        self._parms["score_each_iteration"] = score_each_iteration


    @property
    def fold_assignment(self):
        """
        Cross-validation fold assignment scheme, if fold_column is not specified. The 'Stratified' option will stratify
        the folds based on the response variable, for classification problems.

        One of: ``"auto"``, ``"random"``, ``"modulo"``, ``"stratified"``  (default: ``"auto"``).
        """
        return self._parms.get("fold_assignment")

    @fold_assignment.setter
    def fold_assignment(self, fold_assignment):
        assert_is_type(fold_assignment, None, Enum("auto", "random", "modulo", "stratified"))
        self._parms["fold_assignment"] = fold_assignment


    @property
    def fold_column(self):
        """
        Column with cross-validation fold index assignment per observation.

        Type: ``str``.
        """
        return self._parms.get("fold_column")

    @fold_column.setter
    def fold_column(self, fold_column):
        assert_is_type(fold_column, None, str)
        self._parms["fold_column"] = fold_column


    @property
    def response_column(self):
        """
        Response variable column.

        Type: ``str``.
        """
        return self._parms.get("response_column")

    @response_column.setter
    def response_column(self, response_column):
        assert_is_type(response_column, None, str)
        self._parms["response_column"] = response_column


    @property
    def ignored_columns(self):
        """
        Names of columns to ignore for training.

        Type: ``List[str]``.
        """
        return self._parms.get("ignored_columns")

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        assert_is_type(ignored_columns, None, [str])
        self._parms["ignored_columns"] = ignored_columns


    @property
    def ignore_const_cols(self):
        """
        Ignore constant columns.

        Type: ``bool``  (default: ``True``).
        """
        return self._parms.get("ignore_const_cols")

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        assert_is_type(ignore_const_cols, None, bool)
        self._parms["ignore_const_cols"] = ignore_const_cols


    @property
    def offset_column(self):
        """
        Offset column. This will be added to the combination of columns before applying the link function.

        Type: ``str``.
        """
        return self._parms.get("offset_column")

    @offset_column.setter
    def offset_column(self, offset_column):
        assert_is_type(offset_column, None, str)
        self._parms["offset_column"] = offset_column


    @property
    def weights_column(self):
        """
        Column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the
        dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative
        weights are not allowed. Note: Weights are per-row observation weights and do not increase the size of the data
        frame. This is typically the number of times a row is repeated, but non-integer values are supported as well.
        During training, rows with higher weights matter more, due to the larger loss function pre-factor.

        Type: ``str``.
        """
        return self._parms.get("weights_column")

    @weights_column.setter
    def weights_column(self, weights_column):
        assert_is_type(weights_column, None, str)
        self._parms["weights_column"] = weights_column


    @property
    def stopping_rounds(self):
        """
        Early stopping based on convergence of stopping_metric. Stop if simple moving average of length k of the
        stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)

        Type: ``int``  (default: ``0``).
        """
        return self._parms.get("stopping_rounds")

    @stopping_rounds.setter
    def stopping_rounds(self, stopping_rounds):
        assert_is_type(stopping_rounds, None, int)
        self._parms["stopping_rounds"] = stopping_rounds


    @property
    def stopping_metric(self):
        """
        Metric to use for early stopping (AUTO: logloss for classification, deviance for regression and anonomaly_score
        for Isolation Forest). Note that custom and custom_increasing can only be used in GBM and DRF with the Python
        client.

        One of: ``"auto"``, ``"deviance"``, ``"logloss"``, ``"mse"``, ``"rmse"``, ``"mae"``, ``"rmsle"``, ``"auc"``,
        ``"lift_top_group"``, ``"misclassification"``, ``"aucpr"``, ``"mean_per_class_error"``, ``"custom"``,
        ``"custom_increasing"``  (default: ``"auto"``).
        """
        return self._parms.get("stopping_metric")

    @stopping_metric.setter
    def stopping_metric(self, stopping_metric):
        assert_is_type(stopping_metric, None, Enum("auto", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "lift_top_group", "misclassification", "aucpr", "mean_per_class_error", "custom", "custom_increasing"))
        self._parms["stopping_metric"] = stopping_metric


    @property
    def stopping_tolerance(self):
        """
        Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much)

        Type: ``float``  (default: ``0.001``).
        """
        return self._parms.get("stopping_tolerance")

    @stopping_tolerance.setter
    def stopping_tolerance(self, stopping_tolerance):
        assert_is_type(stopping_tolerance, None, numeric)
        self._parms["stopping_tolerance"] = stopping_tolerance


    @property
    def max_runtime_secs(self):
        """
        Maximum allowed runtime in seconds for model training. Use 0 to disable.

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("max_runtime_secs")

    @max_runtime_secs.setter
    def max_runtime_secs(self, max_runtime_secs):
        assert_is_type(max_runtime_secs, None, numeric)
        self._parms["max_runtime_secs"] = max_runtime_secs


    @property
    def seed(self):
        """
        Seed for pseudo random number generator (if applicable)

        Type: ``int``  (default: ``-1``).
        """
        return self._parms.get("seed")

    @seed.setter
    def seed(self, seed):
        assert_is_type(seed, None, int)
        self._parms["seed"] = seed


    @property
    def distribution(self):
        """
        Distribution function

        One of: ``"auto"``, ``"bernoulli"``, ``"multinomial"``, ``"gaussian"``, ``"poisson"``, ``"gamma"``,
        ``"tweedie"``, ``"laplace"``, ``"quantile"``, ``"huber"``  (default: ``"auto"``).
        """
        return self._parms.get("distribution")

    @distribution.setter
    def distribution(self, distribution):
        assert_is_type(distribution, None, Enum("auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace", "quantile", "huber"))
        self._parms["distribution"] = distribution


    @property
    def tweedie_power(self):
        """
        Tweedie power for Tweedie regression, must be between 1 and 2.

        Type: ``float``  (default: ``1.5``).
        """
        return self._parms.get("tweedie_power")

    @tweedie_power.setter
    def tweedie_power(self, tweedie_power):
        assert_is_type(tweedie_power, None, numeric)
        self._parms["tweedie_power"] = tweedie_power


    @property
    def categorical_encoding(self):
        """
        Encoding scheme for categorical features

        One of: ``"auto"``, ``"enum"``, ``"one_hot_internal"``, ``"one_hot_explicit"``, ``"binary"``, ``"eigen"``,
        ``"label_encoder"``, ``"sort_by_response"``, ``"enum_limited"``  (default: ``"auto"``).
        """
        return self._parms.get("categorical_encoding")

    @categorical_encoding.setter
    def categorical_encoding(self, categorical_encoding):
        assert_is_type(categorical_encoding, None, Enum("auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder", "sort_by_response", "enum_limited"))
        self._parms["categorical_encoding"] = categorical_encoding


    @property
    def quiet_mode(self):
        """
        Enable quiet mode

        Type: ``bool``  (default: ``True``).
        """
        return self._parms.get("quiet_mode")

    @quiet_mode.setter
    def quiet_mode(self, quiet_mode):
        assert_is_type(quiet_mode, None, bool)
        self._parms["quiet_mode"] = quiet_mode


    @property
    def export_checkpoints_dir(self):
        """
        Automatically export generated models to this directory.

        Type: ``str``.
        """
        return self._parms.get("export_checkpoints_dir")

    @export_checkpoints_dir.setter
    def export_checkpoints_dir(self, export_checkpoints_dir):
        assert_is_type(export_checkpoints_dir, None, str)
        self._parms["export_checkpoints_dir"] = export_checkpoints_dir


    @property
    def ntrees(self):
        """
        (same as n_estimators) Number of trees.

        Type: ``int``  (default: ``50``).
        """
        return self._parms.get("ntrees")

    @ntrees.setter
    def ntrees(self, ntrees):
        assert_is_type(ntrees, None, int)
        self._parms["ntrees"] = ntrees


    @property
    def max_depth(self):
        """
        Maximum tree depth.

        Type: ``int``  (default: ``6``).
        """
        return self._parms.get("max_depth")

    @max_depth.setter
    def max_depth(self, max_depth):
        assert_is_type(max_depth, None, int)
        self._parms["max_depth"] = max_depth


    @property
    def min_rows(self):
        """
        (same as min_child_weight) Fewest allowed (weighted) observations in a leaf.

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("min_rows")

    @min_rows.setter
    def min_rows(self, min_rows):
        assert_is_type(min_rows, None, numeric)
        self._parms["min_rows"] = min_rows


    @property
    def min_child_weight(self):
        """
        (same as min_rows) Fewest allowed (weighted) observations in a leaf.

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("min_child_weight")

    @min_child_weight.setter
    def min_child_weight(self, min_child_weight):
        assert_is_type(min_child_weight, None, numeric)
        self._parms["min_child_weight"] = min_child_weight


    @property
    def learn_rate(self):
        """
        (same as eta) Learning rate (from 0.0 to 1.0)

        Type: ``float``  (default: ``0.3``).
        """
        return self._parms.get("learn_rate")

    @learn_rate.setter
    def learn_rate(self, learn_rate):
        assert_is_type(learn_rate, None, numeric)
        self._parms["learn_rate"] = learn_rate


    @property
    def eta(self):
        """
        (same as learn_rate) Learning rate (from 0.0 to 1.0)

        Type: ``float``  (default: ``0.3``).
        """
        return self._parms.get("eta")

    @eta.setter
    def eta(self, eta):
        assert_is_type(eta, None, numeric)
        self._parms["eta"] = eta


    @property
    def sample_rate(self):
        """
        (same as subsample) Row sample rate per tree (from 0.0 to 1.0)

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("sample_rate")

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        assert_is_type(sample_rate, None, numeric)
        self._parms["sample_rate"] = sample_rate


    @property
    def subsample(self):
        """
        (same as sample_rate) Row sample rate per tree (from 0.0 to 1.0)

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("subsample")

    @subsample.setter
    def subsample(self, subsample):
        assert_is_type(subsample, None, numeric)
        self._parms["subsample"] = subsample


    @property
    def col_sample_rate(self):
        """
        (same as colsample_bylevel) Column sample rate (from 0.0 to 1.0)

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("col_sample_rate")

    @col_sample_rate.setter
    def col_sample_rate(self, col_sample_rate):
        assert_is_type(col_sample_rate, None, numeric)
        self._parms["col_sample_rate"] = col_sample_rate


    @property
    def colsample_bylevel(self):
        """
        (same as col_sample_rate) Column sample rate (from 0.0 to 1.0)

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("colsample_bylevel")

    @colsample_bylevel.setter
    def colsample_bylevel(self, colsample_bylevel):
        assert_is_type(colsample_bylevel, None, numeric)
        self._parms["colsample_bylevel"] = colsample_bylevel


    @property
    def col_sample_rate_per_tree(self):
        """
        (same as colsample_bytree) Column sample rate per tree (from 0.0 to 1.0)

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("col_sample_rate_per_tree")

    @col_sample_rate_per_tree.setter
    def col_sample_rate_per_tree(self, col_sample_rate_per_tree):
        assert_is_type(col_sample_rate_per_tree, None, numeric)
        self._parms["col_sample_rate_per_tree"] = col_sample_rate_per_tree


    @property
    def colsample_bytree(self):
        """
        (same as col_sample_rate_per_tree) Column sample rate per tree (from 0.0 to 1.0)

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("colsample_bytree")

    @colsample_bytree.setter
    def colsample_bytree(self, colsample_bytree):
        assert_is_type(colsample_bytree, None, numeric)
        self._parms["colsample_bytree"] = colsample_bytree


    @property
    def max_abs_leafnode_pred(self):
        """
        (same as max_delta_step) Maximum absolute value of a leaf node prediction

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("max_abs_leafnode_pred")

    @max_abs_leafnode_pred.setter
    def max_abs_leafnode_pred(self, max_abs_leafnode_pred):
        assert_is_type(max_abs_leafnode_pred, None, float)
        self._parms["max_abs_leafnode_pred"] = max_abs_leafnode_pred


    @property
    def max_delta_step(self):
        """
        (same as max_abs_leafnode_pred) Maximum absolute value of a leaf node prediction

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("max_delta_step")

    @max_delta_step.setter
    def max_delta_step(self, max_delta_step):
        assert_is_type(max_delta_step, None, float)
        self._parms["max_delta_step"] = max_delta_step


    @property
    def monotone_constraints(self):
        """
        A mapping representing monotonic constraints. Use +1 to enforce an increasing constraint and -1 to specify a
        decreasing constraint.

        Type: ``dict``.
        """
        return self._parms.get("monotone_constraints")

    @monotone_constraints.setter
    def monotone_constraints(self, monotone_constraints):
        assert_is_type(monotone_constraints, None, dict)
        self._parms["monotone_constraints"] = monotone_constraints


    @property
    def score_tree_interval(self):
        """
        Score the model after every so many trees. Disabled if set to 0.

        Type: ``int``  (default: ``0``).
        """
        return self._parms.get("score_tree_interval")

    @score_tree_interval.setter
    def score_tree_interval(self, score_tree_interval):
        assert_is_type(score_tree_interval, None, int)
        self._parms["score_tree_interval"] = score_tree_interval


    @property
    def min_split_improvement(self):
        """
        (same as gamma) Minimum relative improvement in squared error reduction for a split to happen

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("min_split_improvement")

    @min_split_improvement.setter
    def min_split_improvement(self, min_split_improvement):
        assert_is_type(min_split_improvement, None, float)
        self._parms["min_split_improvement"] = min_split_improvement


    @property
    def gamma(self):
        """
        (same as min_split_improvement) Minimum relative improvement in squared error reduction for a split to happen

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("gamma")

    @gamma.setter
    def gamma(self, gamma):
        assert_is_type(gamma, None, float)
        self._parms["gamma"] = gamma


    @property
    def nthread(self):
        """
        Number of parallel threads that can be used to run XGBoost. Cannot exceed H2O cluster limits (-nthreads
        parameter). Defaults to maximum available

        Type: ``int``  (default: ``-1``).
        """
        return self._parms.get("nthread")

    @nthread.setter
    def nthread(self, nthread):
        assert_is_type(nthread, None, int)
        self._parms["nthread"] = nthread


    @property
    def max_bins(self):
        """
        For tree_method=hist only: maximum number of bins

        Type: ``int``  (default: ``256``).
        """
        return self._parms.get("max_bins")

    @max_bins.setter
    def max_bins(self, max_bins):
        assert_is_type(max_bins, None, int)
        self._parms["max_bins"] = max_bins


    @property
    def max_leaves(self):
        """
        For tree_method=hist only: maximum number of leaves

        Type: ``int``  (default: ``0``).
        """
        return self._parms.get("max_leaves")

    @max_leaves.setter
    def max_leaves(self, max_leaves):
        assert_is_type(max_leaves, None, int)
        self._parms["max_leaves"] = max_leaves


    @property
    def min_sum_hessian_in_leaf(self):
        """
        For tree_method=hist only: the mininum sum of hessian in a leaf to keep splitting

        Type: ``float``  (default: ``100``).
        """
        return self._parms.get("min_sum_hessian_in_leaf")

    @min_sum_hessian_in_leaf.setter
    def min_sum_hessian_in_leaf(self, min_sum_hessian_in_leaf):
        assert_is_type(min_sum_hessian_in_leaf, None, float)
        self._parms["min_sum_hessian_in_leaf"] = min_sum_hessian_in_leaf


    @property
    def min_data_in_leaf(self):
        """
        For tree_method=hist only: the mininum data in a leaf to keep splitting

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("min_data_in_leaf")

    @min_data_in_leaf.setter
    def min_data_in_leaf(self, min_data_in_leaf):
        assert_is_type(min_data_in_leaf, None, float)
        self._parms["min_data_in_leaf"] = min_data_in_leaf


    @property
    def sample_type(self):
        """
        For booster=dart only: sample_type

        One of: ``"uniform"``, ``"weighted"``  (default: ``"uniform"``).
        """
        return self._parms.get("sample_type")

    @sample_type.setter
    def sample_type(self, sample_type):
        assert_is_type(sample_type, None, Enum("uniform", "weighted"))
        self._parms["sample_type"] = sample_type


    @property
    def normalize_type(self):
        """
        For booster=dart only: normalize_type

        One of: ``"tree"``, ``"forest"``  (default: ``"tree"``).
        """
        return self._parms.get("normalize_type")

    @normalize_type.setter
    def normalize_type(self, normalize_type):
        assert_is_type(normalize_type, None, Enum("tree", "forest"))
        self._parms["normalize_type"] = normalize_type


    @property
    def rate_drop(self):
        """
        For booster=dart only: rate_drop (0..1)

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("rate_drop")

    @rate_drop.setter
    def rate_drop(self, rate_drop):
        assert_is_type(rate_drop, None, float)
        self._parms["rate_drop"] = rate_drop


    @property
    def one_drop(self):
        """
        For booster=dart only: one_drop

        Type: ``bool``  (default: ``False``).
        """
        return self._parms.get("one_drop")

    @one_drop.setter
    def one_drop(self, one_drop):
        assert_is_type(one_drop, None, bool)
        self._parms["one_drop"] = one_drop


    @property
    def skip_drop(self):
        """
        For booster=dart only: skip_drop (0..1)

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("skip_drop")

    @skip_drop.setter
    def skip_drop(self, skip_drop):
        assert_is_type(skip_drop, None, float)
        self._parms["skip_drop"] = skip_drop


    @property
    def tree_method(self):
        """
        Tree method

        One of: ``"auto"``, ``"exact"``, ``"approx"``, ``"hist"``  (default: ``"auto"``).
        """
        return self._parms.get("tree_method")

    @tree_method.setter
    def tree_method(self, tree_method):
        assert_is_type(tree_method, None, Enum("auto", "exact", "approx", "hist"))
        self._parms["tree_method"] = tree_method


    @property
    def grow_policy(self):
        """
        Grow policy - depthwise is standard GBM, lossguide is LightGBM

        One of: ``"depthwise"``, ``"lossguide"``  (default: ``"depthwise"``).
        """
        return self._parms.get("grow_policy")

    @grow_policy.setter
    def grow_policy(self, grow_policy):
        assert_is_type(grow_policy, None, Enum("depthwise", "lossguide"))
        self._parms["grow_policy"] = grow_policy


    @property
    def booster(self):
        """
        Booster type

        One of: ``"gbtree"``, ``"gblinear"``, ``"dart"``  (default: ``"gbtree"``).
        """
        return self._parms.get("booster")

    @booster.setter
    def booster(self, booster):
        assert_is_type(booster, None, Enum("gbtree", "gblinear", "dart"))
        self._parms["booster"] = booster


    @property
    def reg_lambda(self):
        """
        L2 regularization

        Type: ``float``  (default: ``1``).
        """
        return self._parms.get("reg_lambda")

    @reg_lambda.setter
    def reg_lambda(self, reg_lambda):
        assert_is_type(reg_lambda, None, float)
        self._parms["reg_lambda"] = reg_lambda


    @property
    def reg_alpha(self):
        """
        L1 regularization

        Type: ``float``  (default: ``0``).
        """
        return self._parms.get("reg_alpha")

    @reg_alpha.setter
    def reg_alpha(self, reg_alpha):
        assert_is_type(reg_alpha, None, float)
        self._parms["reg_alpha"] = reg_alpha


    @property
    def dmatrix_type(self):
        """
        Type of DMatrix. For sparse, NAs and 0 are treated equally.

        One of: ``"auto"``, ``"dense"``, ``"sparse"``  (default: ``"auto"``).
        """
        return self._parms.get("dmatrix_type")

    @dmatrix_type.setter
    def dmatrix_type(self, dmatrix_type):
        assert_is_type(dmatrix_type, None, Enum("auto", "dense", "sparse"))
        self._parms["dmatrix_type"] = dmatrix_type


    @property
    def backend(self):
        """
        Backend. By default (auto), a GPU is used if available.

        One of: ``"auto"``, ``"gpu"``, ``"cpu"``  (default: ``"auto"``).
        """
        return self._parms.get("backend")

    @backend.setter
    def backend(self, backend):
        assert_is_type(backend, None, Enum("auto", "gpu", "cpu"))
        self._parms["backend"] = backend


    @property
    def gpu_id(self):
        """
        Which GPU to use.

        Type: ``int``  (default: ``0``).
        """
        return self._parms.get("gpu_id")

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        assert_is_type(gpu_id, None, int)
        self._parms["gpu_id"] = gpu_id


    @staticmethod
    def available():
        """
        Ask the H2O server whether a XGBoost model can be built (depends on availability of native backends).
        :return: True if a XGBoost model can be built, or False otherwise.
        """
        if "XGBoost" not in h2o.cluster().list_core_extensions():
            print("Cannot build an XGBoost model - no backend found.")
            return False
        else:
            return True
