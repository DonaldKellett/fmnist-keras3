from src.data import FashionMNIST
from src.model import SoftmaxRegression
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-model', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    data = FashionMNIST(num_categories=10)
    model = SoftmaxRegression(data, lr=0.75)
    history = model.fit_data()
    loss, accuracy = model.evaluate_data()
    print(f'Validation accuracy: {accuracy}')

    if args.save_model:
        model.save_model('model.keras')
