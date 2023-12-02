import keras

class SoftmaxRegression:
    def __init__(self, data, sigma=0.01, lr=0.1, batch_size=64, epochs=10):
        self.data = data
        self.sigma = sigma
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = keras.Sequential([
            keras.layers.Input(shape=(self.data.height, self.data.width, self.data.channels)),
            keras.layers.Flatten(),
            keras.layers.UnitNormalization(),
            keras.layers.Dense(
                self.data.num_categories,
                activation='softmax',
                kernel_initializer=keras.initializers.RandomNormal(stddev=self.sigma)
            )
        ])
        self.model.compile(
            optimizer=keras.optimizers.SGD(self.lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def fit_data(self):
        return self.model.fit(
            self.data.X_train,
            self.data.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs
        )
    
    def evaluate_data(self):
        return self.model.evaluate(
            self.data.X_val,
            self.data.y_val,
            batch_size=self.batch_size
        )
    
    def save_model(self, path):
        self.model.save(path)
