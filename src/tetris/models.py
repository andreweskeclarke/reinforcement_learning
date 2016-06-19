import tetris_theano


def compile(model_name):
    return {
        'complex': complex_model,
        'dqn': dqn_model,
        'imagenet': imagenet_model
    }[model_name]()

def complex_model():
    layer1_input = 128 * (12) * (2)
    model = tetris_theano.Model([
            tetris_theano.Conv2DLayer(32, 3, 3, 1, 10, 20),
            tetris_theano.Conv2DLayer(32, 3, 3, 32, 8, 18),
            tetris_theano.Conv2DLayer(64, 3, 3, 32, 6, 16),
            tetris_theano.Conv2DLayer(128, 2, 2, 64, 4, 14),
            tetris_theano.Conv2DLayer(128, 2, 2, 128, 3, 13),
            tetris_theano.Flatten(),
            tetris_theano.Split([tetris_theano.DenseLayer(layer1_input, 256),
                                 tetris_theano.DenseLayer(256, len(POSSIBLE_MOVES))],
                                [tetris_theano.DenseLayer(layer1_input, 256),
                                 tetris_theano.DenseLayer(256, 1)],
                                tetris_theano.ActionAdvantageMerge())
        ])
    model.compile()
    return model

def dqn_model():
    layer1_input = 128 * (12) * (2)
    model = tetris_theano.Model([
            tetris_theano.Conv2DLayer(32, 4, 4, 2, 10, 20),
            tetris_theano.Flatten(),
        ])
    model.compile()
    return model
