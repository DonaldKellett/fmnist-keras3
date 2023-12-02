# fmnist-keras3

[Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/) linear classification model with [Keras 3.0](https://keras.io/) and [PyTorch](https://pytorch.org/)

## Quickstart

Just install Python 3.11 - no GPU or specialized hardware required.

We recommend [pyenv](https://github.com/pyenv/pyenv) for creating and managing virtual environments.

Install project dependencies:

```bash
pip install -r requirements.txt
```

Now train the model:

```bash
python main.py
```

Optionally save the model with `--save-model`:

```bash
python main.py --save-model
```

Run the unit tests:

```bash
pytest -v
```

## Credits

Adapted from Ch. 4.2-4.5 of [D2L.ai](http://d2l.ai/) for the latest and greatest [Keras 3.0](https://keras.io/) framework with [PyTorch](https://pytorch.org/) as the backend

## License

[MIT](./LICENSE)
