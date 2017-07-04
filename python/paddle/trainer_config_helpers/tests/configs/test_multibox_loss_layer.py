from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

input_loc = data_layer(name='input_loc', size=16, height=16, width=1)

input_conf = data_layer(name='input_conf', size=8, height=1, width=8)

priorbox = data_layer(name='priorbox', size=32, height=4, width=8)

label = data_layer(name='label', size=24, height=4, width=6)

multibox_loss = multibox_loss_layer(
    input_loc=input_loc,
    input_conf=input_conf,
    priorbox=priorbox,
    label=label,
    num_classes=21,
    overlap_threshold=0.5,
    neg_pos_ratio=3.0,
    neg_overlap=0.5,
    background_id=0,
    name='test_multibox_loss')

outputs(multibox_loss)
