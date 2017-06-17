import tetris_theano
from tetris_game import BOARD_HEIGHT, BOARD_WIDTH, POSSIBLE_MOVES


def compile(model_name):
    return {
        'action_advantage': action_advantage,
        'dqn': dqn,
        'dqn_piece_prediction': dqn_piece_prediction,
        'single_layer_piece_prediction': single_layer_piece_prediction,
        'super_deep_piece_prediction': super_deep_piece_prediction,
        'dense_piece_prediction': dense_piece_prediction,
        'dense_value_prediction': dense_value_prediction,
        'simple_value_prediction': simple_value_prediction,
    }[model_name]()


def action_advantage():
    layer1_input = 128 * (12) * (2)
    model = tetris_theano.Model([
        tetris_theano.Conv2DLayer(32, 3, 3, 1, BOARD_WIDTH, BOARD_HEIGHT),
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


def dqn():
    conv_output_size = (BOARD_HEIGHT - 4) * (BOARD_WIDTH - 4) * 32 * 16
    model = tetris_theano.Model([
        tetris_theano.Conv2DLayer(16, 3, 3, 1, BOARD_WIDTH, BOARD_HEIGHT),
        tetris_theano.Conv2DLayer(32, 3, 3, 1, BOARD_WIDTH - 2, BOARD_HEIGHT - 2),
        tetris_theano.Flatten(),
        tetris_theano.DenseLayer(conv_output_size, 256),
        tetris_theano.DenseLayer(256, len(POSSIBLE_MOVES))
    ])
    model.compile()
    return model


def single_layer_piece_prediction():
    layer1_input = BOARD_WIDTH * BOARD_HEIGHT + len(POSSIBLE_MOVES)
    model = tetris_theano.Model([
        tetris_theano.StateAndActionMerge(),
        tetris_theano.DenseLayer(layer1_input, 256),
        tetris_theano.DenseLayer(256, BOARD_WIDTH * BOARD_HEIGHT)
    ])
    model.compile()
    return model


def dense_value_prediction():
    layer1_input = BOARD_HEIGHT * BOARD_WIDTH
    model = tetris_theano.Model([
        tetris_theano.Flatten(),
        tetris_theano.DenseLayer(layer1_input, 1024),
        tetris_theano.DenseLayer(1024, 1024),
        tetris_theano.DenseLayer(1024, 512),
        tetris_theano.DenseLayer(512, 512),
        tetris_theano.DenseLayer(512, 512),
        tetris_theano.DenseLayer(512, 1)
    ])
    model.compile()
    return model


def simple_value_prediction():
    layer1_input = BOARD_HEIGHT * BOARD_WIDTH
    model = tetris_theano.Model([
        tetris_theano.Flatten(),
        tetris_theano.DenseLayer(layer1_input, 1024),
        tetris_theano.DenseLayer(1024, 256),
        tetris_theano.DenseLayer(256, 1),
    ])
    model.compile()
    return model


def dense_piece_prediction():
    layer1_input = len(POSSIBLE_MOVES) + BOARD_HEIGHT * BOARD_WIDTH
    model = tetris_theano.Model([
        tetris_theano.Flatten(),
        tetris_theano.StateAndActionMerge(),
        tetris_theano.DenseLayer(layer1_input, 1024),
        tetris_theano.DenseLayer(1024, 1024),
        tetris_theano.DenseLayer(1024, 512),
        tetris_theano.DenseLayer(512, 512),
        tetris_theano.DenseLayer(512, 512),
        tetris_theano.DenseLayer(512, BOARD_WIDTH * BOARD_HEIGHT)
    ])
    model.compile()
    return model


def dqn_piece_prediction():
    conv_output_size = (BOARD_HEIGHT - 4) * (BOARD_WIDTH - 4) * 32 * 16
    layer1_input = len(POSSIBLE_MOVES) + conv_output_size
    model = tetris_theano.Model([
        tetris_theano.Conv2DLayer(16, 3, 3, 1, BOARD_WIDTH, BOARD_HEIGHT),
        tetris_theano.Conv2DLayer(32, 3, 3, 1, BOARD_WIDTH - 2, BOARD_HEIGHT - 2),
        tetris_theano.Flatten(),
        tetris_theano.StateAndActionMerge(),
        tetris_theano.DenseLayer(layer1_input, 256),
        tetris_theano.DenseLayer(256, BOARD_WIDTH * BOARD_HEIGHT)
    ])
    model.compile()
    return model


def super_deep_piece_prediction():
    conv_output_size = 2 * 12 * 128
    layer1_input = len(POSSIBLE_MOVES) + conv_output_size
    model = tetris_theano.Model([
        tetris_theano.Conv2DLayer(64, 3, 3, 1, BOARD_WIDTH, BOARD_HEIGHT),
        tetris_theano.Conv2DLayer(128, 3, 3, 1, 8, 18),
        tetris_theano.Conv2DLayer(256, 3, 3, 1, 6, 16),
        tetris_theano.Conv2DLayer(128, 3, 3, 1, 4, 14),
        tetris_theano.Flatten(),
        tetris_theano.StateAndActionMerge(),
        tetris_theano.DenseLayer(layer1_input, 512),
        tetris_theano.DenseLayer(512, 512),
        tetris_theano.DenseLayer(256, BOARD_WIDTH * BOARD_HEIGHT)
    ])
    model.compile()
    return model
