import argparse
import mxnet as mx
import json
import os
import logging


head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


def parse_args():
    """
    Parses input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--symbol', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--begin_epoch', type=int)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--train_rec', type=str)
    parser.add_argument('--val_rec', type=str)
    parser.add_argument('--model', type=str)

    return parser.parse_args()


def get_iterators(train_rec, val_rec, batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = train_rec,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True
    )
    val = mx.io.ImageRecordIter(
        path_imgrec         = val_rec,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False
    )

    return (train, val)


def get_symbol(symbol, num_classes, is_train=True):
    body = mx.symbol.load(symbol)
    internals = body.get_internals()
    d = internals.list_outputs()
    arg_params, out, aux_params = internals.infer_shape(data=(1,3,224,224))
    print(list(zip(d, out)))

    relu = internals['relu5_3_output']
    pool = mx.symbol.Pooling(data=relu, pool_type='max', kernel=(13,13), stride=(13,13), name='pool')
    flat = mx.symbol.Flatten(data=pool)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=num_classes, name='fc2')
    net = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    return net


# def fit(symbol, train, val, batch_size, num_gpus, begin_epoch, num_epoch, params):
#     devs = [mx.gpu(i) for i in range(num_gpus)]
#     mod = mx.mod.Module(symbol=symbol, context=devs)
#     mod.fit(train, val,
#         arg_params=None,
#         aux_params=None,
#         allow_missing=True,
#         batch_end_callback = mx.callback.Speedometer(batch_size, 10),
#         kvstore='device',
#         optimizer='sgd',
#         begin_epoch=begin_epoch,
#         num_epoch=num_epoch,
#         optimizer_params={'learning_rate':0.05},
#         initializer=mx.init.Load(params, mx.init.Xavier(), verbose=True),
#         # initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
#         eval_metric='acc')
#     metric = mx.metric.Accuracy()
#     return mod.score(val, metric)


def main():
    args = parse_args()
    symbol = args.symbol
    num_classes = args.num_classes
    num_gpus = args.num_gpus
    train_rec = args.train_rec
    val_rec = args.val_rec
    batch_size = args.batch_size
    params = args.params
    begin_epoch = args.begin_epoch
    num_epoch = args.num_epoch
    model = args.model


    new_sym = get_symbol(symbol, num_classes, is_train=False)

    train_iter, val_iter = get_iterators(train_rec, val_rec, batch_size)

    # fit(new_sym, train_iter, val_iter, batch_size, num_gpus, begin_epoch, num_epoch, params)

    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=new_sym, context=devs)
    mod.fit(train_iter, val_iter,
        arg_params=None,
        aux_params=None,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        epoch_end_callback= mx.callback.do_checkpoint(model),
        kvstore='device',
        optimizer='sgd',
        begin_epoch=begin_epoch,
        num_epoch=num_epoch,
        optimizer_params={'learning_rate':0.05},
        initializer=mx.init.Load(params, mx.init.Xavier(), verbose=True),
        # initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')


if __name__ == '__main__':
    main()