from copy import deepcopy

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict


class MLPClassWrapper(MLPClassifier):

    def __init__(self, hidden_layer_dimension=1, hidden_layer_value=100, activation="relu", *, solver="adam",
                 alpha=0.0001, batch_size="auto", learning_rate="constant", learning_rate_init=0.001, power_t=0.5,
                 max_iter=10000, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000, **kwargs):
        self._hidden_layer_dimension = hidden_layer_dimension
        self._hidden_layer_value = hidden_layer_value
        super().__init__(hidden_layer_sizes=(hidden_layer_value,) * hidden_layer_dimension, activation=activation,
                         solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle,
                         random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum,
                         nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                         validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                         n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    @property
    def hidden_layer_dimension(self):
        return self._hidden_layer_dimension

    @property
    def hidden_layer_value(self):
        return self._hidden_layer_value

    @hidden_layer_dimension.setter
    def hidden_layer_dimension(self, value):
        self._hidden_layer_dimension = value
        self.hidden_layer_sizes = [self.hidden_layer_value] * self.hidden_layer_dimension

    @hidden_layer_value.setter
    def hidden_layer_value(self, value):
        self._hidden_layer_value = value
        self.hidden_layer_sizes = [self.hidden_layer_value] * self.hidden_layer_dimension

    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


class BaseAdaBoost(AdaBoostClassifier):

    def __repr__(self):
        return self.c_base_estimator.__repr__()

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            if hasattr(self, key):
                value = getattr(self, key)
            elif hasattr(self.c_base_estimator, key):
                value = getattr(self.c_base_estimator, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True) | self.c_base_estimator.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                if hasattr(self, key):
                    setattr(self, key, value)
                elif hasattr(self.c_base_estimator, key):
                    setattr(self.c_base_estimator, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class DTAdaBoost(BaseAdaBoost):

    def __init__(self, criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, n_estimators=50, learning_rate=1.0,
                 algorithm="SAMME.R"):
        self.c_base_estimator = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                       max_features=max_features, random_state=random_state,
                                                       max_leaf_nodes=max_leaf_nodes,
                                                       min_impurity_decrease=min_impurity_decrease,
                                                       class_weight=class_weight,
                                                       ccp_alpha=ccp_alpha)
        super().__init__(base_estimator=self.c_base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                         algorithm=algorithm, random_state=random_state)


class SVCAdaBoost(BaseAdaBoost):

    def __init__(self, n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=None, C=1.0, kernel="rbf",
                 degree=3, gamma="scale", coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovr", break_ties=False):
        self.c_base_estimator = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                                    probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                                    verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                                    break_ties=break_ties, random_state=random_state)
        super().__init__(base_estimator=self.c_base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                         algorithm=algorithm, random_state=random_state)


class VotingAdaBoost(BaseAdaBoost):

    def __init__(self, n_estimators=10, learning_rate=0.2, algorithm="SAMME.R", random_state=None, mlp_w=1., svc_w=1., dt_w=1.):

        self.voting_estimators = [
            ("MLPClassWrapper",
             MLPClassWrapper(hidden_layer_dimension=2, hidden_layer_value=3, activation="logistic", solver="adam",
                             alpha=0.0001, learning_rate="adaptive", learning_rate_init=0.0015)),
            ("SVC", SVC(C=4.0, kernel="linear", gamma="scale", probability=True, tol=0.0008)),
            ("DecisionTreeClassifier",
             DecisionTreeClassifier(criterion="log_loss", splitter="best", max_depth=10, min_samples_split=20,
                                    min_samples_leaf=3, max_features=20, max_leaf_nodes=350, ccp_alpha=0.005))
        ]

        self.mlp_w = mlp_w
        self.svc_w = svc_w
        self.dt_w = dt_w
        self.c_base_estimator = VotingClassifier(estimators=self.voting_estimators, voting="soft",
                                               weights=(mlp_w, svc_w, dt_w))

        super().__init__(base_estimator=self.c_base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                         algorithm=algorithm, random_state=random_state)
