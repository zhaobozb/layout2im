import os
import re
import torch
from pathlib import Path


def prepare_dir(name):
    log_save_dir = 'checkpoints/{}/logs'.format(name)
    model_save_dir = 'checkpoints/{}/models'.format(name)
    sample_save_dir = 'checkpoints/{}/samples'.format(name)
    result_save_dir = 'checkpoints/{}/results'.format(name)

    if not Path(log_save_dir).exists(): Path(log_save_dir).mkdir(parents=True)
    if not Path(model_save_dir).exists(): Path(model_save_dir).mkdir(parents=True)
    if not Path(sample_save_dir).exists(): Path(sample_save_dir).mkdir(parents=True)
    if not Path(result_save_dir).exists(): Path(result_save_dir).mkdir(parents=True)

    return log_save_dir, model_save_dir, sample_save_dir, result_save_dir


def load_model(model, model_dir=None, appendix=None, iter='l'):

    load_iter = None
    load_model = None

    if iter == 's' or not os.path.isdir(model_dir) or len(os.listdir(model_dir)) == 0:
        load_iter = 0
        if not os.path.isdir(model_dir):
            print('models dir not exist')
        elif len(os.listdir(model_dir)) == 0:
            print('models dir is empty')

        print('train from scratch.')
        return load_iter

    # load latest epoch
    if iter == 'l':
        for file in os.listdir(model_dir):
            if appendix is not None and appendix not in file:
                continue

            if file.endswith('.pkl'):
                current_iter = re.search('iter-\d+', file).group(0).split('-')[1]

                if len(current_iter) > 0:
                    current_iter = int(current_iter)

                    if load_iter is None or current_iter > load_iter:
                        load_iter = current_iter
                        load_model = os.path.join(model_dir, file)
                else:
                    continue

        print('load from iter: %d' % load_iter)
        model.load_state_dict(torch.load(load_model))

        return load_iter
    # from given iter
    else:
        iter = int(iter)
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                current_iter = re.search('iter-\d+', file).group(0).split('-')[1]
                if len(current_iter) > 0:
                    if int(current_iter) == iter:
                        load_iter = iter
                        load_model = os.path.join(model_dir, file)
                        break
        if load_model:
            model.load_state_dict(torch.load(load_model))
            print('load from iter: %d' % load_iter)
        else:
            load_iter = 0
            print('there is not saved models of iter %d' % iter)
            print('train from scratch.')
        return load_iter


def save_model(model, model_dir=None, appendix=None, iter=1, save_num=5, save_step=1000):
    iter_idx = range(iter, iter - save_num * save_step, -save_step)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for file in os.listdir(model_dir):
        if file.endswith('.pkl'):
            current_iter = re.search('iter-\d+', file).group(0).split('-')[1]
            if len(current_iter) > 0:
                if int(current_iter) not in iter_idx:
                    os.remove(os.path.join(model_dir, file))
            else:
                continue

    if appendix:
        model_name = os.path.join(model_dir, 'iter-%d_%s.pkl' % (iter, appendix))
    else:
        model_name = os.path.join(model_dir, 'iter-%d.pkl' % iter)
    torch.save(model.state_dict(), model_name)
