# coding: utf-8
'''自动调节单一超参数脚本'''
import json, logging, os, collections, re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-fn', type=str, nargs='?', const='cws' ,default='ngrams')
fn = parser.parse_args().fn
args = ['-tes']

log, cmd = '%s.log' % fn, 'python %s.py' % fn
cmd += ' %s > ' + log
logging.basicConfig(format='%(asctime)s [line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log + 's',
                    filemode='w')

param, best = collections.OrderedDict(), collections.OrderedDict()
if fn != 'cws': 
    param['univ'] = []; param['ngs'] = [2, 3, 4, 1]  # args.append('-min 5 -div 1000 -reg 0 -emb 256 -ff 512 -bn -dropout 0.2 -ngs 4'); 
    # param['cor'] = ['pku', 'msr', 'as', 'cityu']
# param['min'] = [2, 3, 5, 10]
param['bat'] = [128, 64] # , 256, 512, 1024]
param['emb'] = [512, 256, 1024]  # , 128, 2048]
# param['nop'] = []
# param['div'] = [10000, 100, 1000, 100000]
param['ff'] = [2048, 1024] # , 512, 4096]
param['reg'] = [2e-6, 3e-6, 5e-6] # 1e-6, 5e-7, 5e-8, 1e-5, 5e-5, 1e-4]
param['bias_reg'] = [0, 1e-4, 5e-4] # , 1e-7, 1e-6, 1e-5]
param['embr'] = [0, 1e-8] # , 1e-7, 5e-8, 1e-6, 5e-6, 1e-5]
param['dropout'] = [0.5, 0.2, 0.1] # , 0.6, 0.7, 0.8, 0.4, 0.3]
# param['embd'] = []
param['head'] = [8, 16] # , 4, 2, 1]
# param['lr'] = [5e-4, 1e-3, 3e-3, 1e-4, 5e-5]
# param['resn'] = []
param['bn'] = []
# param['head_size'] = [0, 256, 512, 1024, 2048]
param['quad'] = []
# param['rec'] = []
# param['sub'] = []
# param['mask'] = []
# param['shuf'] = []

cnt = 1
command = cmd % ' '.join(args)
logging.info(command); os.system(command)
with open(log, 'rb') as f: f1s = re.findall('F MEASURE:\s+(0\.\d+)', f.read())
max_f1 = max(map(float, f1s))
logging.info('result %d: f1: %s; %d epoches, f1 for each: %s\n', cnt, max_f1, len(f1s), ' '.join(f1s))

for par, ps in param.items():
    max_id = 0
    if not ps:
        cnt += 1
        new_args = ['-' + par]
        command = cmd % ' '.join(args + new_args)
        logging.info(command); os.system(command)
        with open(log, 'rb') as f: f1s = re.findall('F MEASURE:\s+(0\.\d+)', f.read())
        if f1s: f1 = max(map(float, f1s))
        if f1 > max_f1: max_f1, max_id = f1, i; args += new_args; best[par] = bool(max_id)
        logging.info('result %d: Arg: %s; best: %s; f1: %s, bf1: %s; %d epoches, f1 for each: %s\n', 
                      cnt, par, bool(max_id), f1, max_f1, len(f1s), ' '.join(f1s))
    else:
        for i, p in enumerate(ps[1:]):
            cnt += 1; i += 1
            new_args = ['-' + par, str(p)]
            command = cmd % ' '.join(args + new_args)
            logging.info(command); os.system(command)
            with open(log, 'rb') as f: f1s = re.findall('F MEASURE:\s+(0\.\d+)', f.read())
            if f1s: f1 = max(map(float, f1s))
            if f1 > max_f1: max_f1, max_id = f1, i
            logging.info('result %d: Arg: %s; now: %s, best: %s; f1: %s, bf1: %s; %d epoches, f1 for each: %s\n', 
                          cnt, par, p, ps[max_id], f1, max_f1, len(f1s), ' '.join(f1s))
        if max_id: args += ['-' + par, str(ps[max_id])]
        best[par] = ps[max_id]

logging.info('Best args: %s\n%s', ' '.join(args), json.dumps(best))
