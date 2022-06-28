from sklearn.neural_network import MLPClassifier


class MLPClassWrapper(MLPClassifier):

    def __init__(self, hidden_layer_dimension=1, hidden_layer_value=100, activation="relu", *, solver="adam", alpha=0.0001, batch_size="auto", learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000, **kwargs):
        self._hidden_layer_dimension = hidden_layer_dimension
        self._hidden_layer_value = hidden_layer_value
        super().__init__(hidden_layer_sizes=(hidden_layer_value, ) * hidden_layer_dimension, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

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
