from unittest import TestCase
from src.data import FashionMNIST
from src.model import SoftmaxRegression

class TestModel(TestCase):
    def setUp(self) -> None:
        self.data = FashionMNIST(num_categories=10)
        self.model = SoftmaxRegression(self.data, lr=0.75)
        self.history = self.model.fit_data()
    
    def test_model(self) -> None:
        loss, accuracy = self.model.evaluate_data()
        self.assertGreaterEqual(accuracy, 0.8)
