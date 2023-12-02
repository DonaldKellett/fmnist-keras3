import keras
import keras.ops as np

class FashionMNIST:
    def __init__(self, num_categories):
        self.num_categories = num_categories
        (self.X_train, self.y_train), (self.X_val, self.y_val) = keras.datasets.fashion_mnist.load_data()
        self.height = self.X_train.shape[1]
        self.width = self.X_train.shape[2]
        self.channels = 1
        self.X_train = np.expand_dims(self.X_train, -1)
        self.X_val = np.expand_dims(self.X_val, -1)
