# coding: utf-8
'''多层多元组纯注意力分词模型'''
import os, re, keras, collections
import numpy as np

punc_cn = u'＂＃＄＆＇（）＊＋，：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿‘‛“”„‟…‧﹏﹑﹔！？｡。' # ·．–—％’／－
sub_strs = collections.OrderedDict()
sub_strs['nums'] = (u'[－\-＋+]?[0-9０-９]+(?:[.·．][0-9０-９]+)?[%％]?', '0')  # [－\-＋+’]?
sub_strs['nums_cn'] = (u'[零○一二三四五六七八九十百千万亿萬億]{2,}', '9')  # u'[壹贰叁肆伍陆柒捌玖拾佰仟]{2,}'
sub_strs['eng'] = (u'[Ａ-ＺA-Zａ-ｚa-z_.]+', 'a')

def pre(path, inf, encoding='utf8', sub=True, char_idx=None, lb_idx=None, bi=True, ngs_num=0,
        test_set=False, min_count=2, batch_size=32, max_seq_len=75, shuffle=True, smooth=0, rev=True):
    '''文本字符串预处理'''
    with open(os.path.join(path, inf), 'rb') as f:
        corpus = f.read().decode(encoding).replace(u'\u3000', ' ')
    sub_list, lines, X, xlens, Y, ngrams = {'puncs': []}, [], [], [], [], [[] for n in range(ngs_num)]
    if sub:
        for strs in sub_strs:
            if test_set: sub_list[strs] = re.findall(sub_strs[strs][0], corpus)  # re.U
            corpus = re.sub(sub_strs[strs][0], sub_strs[strs][1], corpus)
    if test_set:
        bat_idx = []
        for i, line in enumerate(corpus.split('\n')):
            # puncs_list, pcl = re.findall('[%s\s]+' % punc_cn, line), []
            # for puncs in puncs_list:
            #     pcl.append(u' —— '.join([' '.join(list(punc)) for punc in puncs.strip().split(u'——')]))
            # sub_list['puncs'].append(pcl)
            sub_list['puncs'].append([' '.join([p for p in ps if p.strip()])for ps in re.findall('[%s\s]+' % punc_cn, line)])
            sents = re.split('[%s\s]+' % punc_cn, line)
            lines.append(sents)
            bat_idx += [(i, j) for j in range(len(sents))]
        assert char_idx and lb_idx, 'For test set, char and label id mapping must be given!'
        bat_idx.sort(key=lambda x: -1 * len(lines[x[0]][x[1]]))
        k, bat_x, lens, ngs = 0, [], [], [[] for n in range(ngs_num)]; i, j = bat_idx[k]; sent = lines[i][j]; max_len = len(sent)
        while len(sent) > 1:
            x, ng, pad = [], [[] for n in range(ngs_num)], [0]*(max_len - len(sent))
            last = u'＄'*ngs_num if rev else u'＾'*ngs_num
            if rev: sent  = sent[::-1]
            for c in sent:
                x.append(char_idx.get(c, 1))        # 未登录字id为1
                for n in range(ngs_num):
                    g = c + last[:n+1] if rev else last[-1*(n+1):] + c
                    ng[n].append(char_idx.get(g, n + 2))  # 未登录n元组为n+1
                last = c + last[:-1] if rev else last[1:] + c
            # bat_x.append([char_idx.get(c, 1) for c in lines[i][j]] + [0]*(max_len-len(lines[i][j])))
            for n in range(ngs_num): ngs[n].append(ng[n][::-1] + pad if rev else ng[n] + pad)
            bat_x.append(x[::-1] + pad if rev else x + pad); lens.append(len(sent))
            k += 1; i, j = bat_idx[k]; sent = lines[i][j]
            if len(bat_x) == batch_size:
                X.append(bat_x); xlens.append(lens)
                for n in range(ngs_num): ngrams[n].append(ngs[n])
                bat_x, lens, ngs, max_len = [], [], [[] for n in range(ngs_num)], len(sent)
        if bat_x: 
            X.append(bat_x); xlens.append(lens)
            for n in range(ngs_num): ngrams[n].append(ngs[n])
        return X, xlens, ngrams, lines, sub_list, bat_idx

    labels, cnts = [], {}  # 第一次遍历训练语料，分行、分句并统计字频
    for sent in re.split('[%s\r\n]+' % punc_cn, corpus):
        if not sent.strip(): continue
        chars, lbs = [], []
        last = u'＄'*ngs_num if rev else u'＾'*ngs_num
        words = sent.split()[::-1] if rev else sent.split()
        for word in words:
            lbs.append('s' + 'a'*(len(word)-1) if bi else 's' if len(word) == 1 else 'b%se' % ('m'*(len(word)-2)))
            chars.append(word)
            if rev: word = word[::-1]
            for c in word: 
                cnts[c] = cnts.get(c, 0) + 1
                for n in range(ngs_num):
                    ng = c + last[:n+1] if rev else last[-1*(n+1):] + c
                    cnts[ng] = cnts.get(ng, 0) + 1
                last = c + last[:-1] if rev else last[1:] + c
            if max_seq_len and sum(map(len, chars)) > max_seq_len:
                lines.append(''.join(chars[::-1] if rev else chars[::-1]))
                labels.append(''.join(lbs[::-1] if rev else lbs[::-1]))
                chars, lbs = [], []
        lines.append(''.join(chars[::-1] if rev else chars[::-1]))
        labels.append(''.join(lbs[::-1] if rev else lbs[::-1]))

    if not char_idx:
        i, char_idx = ngs_num + 2, {}   # cnts = sorted(cnts.items(), key=lambda x: x[1], reversed=True)
        for c in cnts:
            if cnts[c] >= min_count: char_idx[c] = i; i += 1
    if not lb_idx:
        if bi:
            lb_idx = {s: abs(i-smooth) for i, s in enumerate('sa')} if smooth else {s: i for i, s in enumerate('sa')}
        else: lb_idx = {s: i+1 for i, s in enumerate('bmes')}

    bat_idx = sorted(list(range(len(lines))), key=lambda x: -1 * len(lines[x]))  # 第二次遍历语料，按句子长度排序、分batch，映射为id
    bat_num = len(lines)//batch_size+1 if len(lines) % batch_size else len(lines)//batch_size
    for i in range(bat_num):
        st, end = i*batch_size, (i+1)*batch_size if (i+1)*batch_size <= len(lines) else -1
        max_len = len(lines[bat_idx[st]])
        if not max_len: break
        bat_x, bat_y, lens, ngs = [], [], [], [[] for n in range(ngs_num)]
        for j in bat_idx[st: end]:
            l, x, y, ng = len(lines[j]), [], [], [[] for n in range(ngs_num)]; pad = [0] * (max_len - l)
            last = u'＄'*ngs_num if rev else u'＾'*ngs_num
            for k in range(l):
                x.append(char_idx.get(lines[j][k], 1))
                y.append(lb_idx[labels[j][k]])
                c = lines[j][-1*(k+1)] if rev else lines[j][k]
                for n in range(ngs_num):
                    g = c + last[:n+1] if rev else last[-1*(n+1):] + c
                    ng[n].append(char_idx.get(g, n + 2))
                last = c + last[:-1] if rev else last[1:] + c
            bat_x.append(x + pad); bat_y.append(y + pad); lens.append(l)
            for n in range(ngs_num): ngs[n].append(ng[n][::-1] + pad if rev else ng[n] + pad)
        X.append(bat_x); Y.append(bat_y); xlens.append(lens)
        for n in range(ngs_num): ngrams[n].append(ngs[n])
        # X.append([[char_idx.get(c, 1) for c in lines[k]] + [0]*(max_len-len(lines[k])) for k in bat_idx[st: end]])
        # Y.append([[lb_idx[c] for c in labels[k]] + [0]*(max_len-len(labels[k])) for k in bat_idx[st: end]])
        # xlens.append([len(lines[k]) for k in bat_idx[st: end]])
    if shuffle:
        sfl = np.arange(len(X)); np.random.shuffle(sfl)
        X = [X[i] for i in sfl]; Y = [Y[i] for i in sfl]; xlens = [xlens[i] for i in sfl]
        for n in range(ngs_num): ngrams[n] = [ngrams[n][i] for i in sfl]
    return X, Y, xlens, ngrams, char_idx, lb_idx, cnts

def inference():
    '''测试集预测阶段处理'''
    print('Model begins testing...')
    Y, sub_chars, sub_cnt = [], {sub_strs[strs][1]: strs for strs in sub_strs}, {sub_strs[strs][1]: 0 for strs in sub_strs}
    new_lines = [['' if len(sent) > 1 else sent for sent in line] for line in lines]
    if not bi: sep_pre, sep_post = {lb_idx[lb] for lb in 'bs'}, {lb_idx[lb] for lb in 'es'}; sep_pre.add(0)
    for i, (bat_x, lens) in enumerate(tqdm(zip(X, xlens))):
        bat_y = model.predict_on_batch([np.array(bat_x), np.array(lens)] + [np.array(ngrams[n][i]) for n in range(ngs_num)])
        if ngs_num and mo: bat_y = np.average(bat_y[avg:], axis=0) if avg >= 0 else bat_y[avg]  # labels list
        for sent_lbs in bat_y:
            Y.append(sent_lbs if bi else list(map(np.argmax, sent_lbs)))
    for (i, j), sent_lbs in zip(bat_idx, Y):
        sent = ''
        for c, lb in zip(lines[i][j], sent_lbs):
            if (bi and lb <= 0.5) or (not bi and lb in sep_pre): sent += ' '
            sent += c
            if not bi and lb in sep_post: sent += ' '
        new_lines[i][j] = sent
    
    for i in range(len(new_lines)):
        j, line = 1, new_lines[i][0].strip()  # lines[i].pop(0).strip()
        while j < len(new_lines[i]):
            line += ' ' + sub_list['puncs'][i][j-1] + ' ' + new_lines[i][j].strip(); j += 1 
            # ' ' + sub_list['puncs'][i].pop(0) + ' ' + lines[i].pop(0).strip() 
        new_lines[i] = line
        # if sub:
        #     line, new_lines[i], j = list(line.strip()), '', 0
        #     while j < len(line):
        #         c = line[j]
        #         new_lines[i] += sub_list[sub_chars[c]].pop(0) if c in sub_chars else c
        # else: new_lines[i] = line
    res = '\n'.join(new_lines)
    if sub:
        k, new_res = 0, ''
        while k < len(res):
            c = res[k]
            if c in sub_chars:
                new_res += sub_list[sub_chars[c]][sub_cnt[c]]  
                sub_cnt[c] += 1
            else: new_res += c
            k += 1
        res = new_res
    with open(outf, 'w') as f:
        f.write(res.encode('utf-8'))  # '\n'.join(new_lines)


if __name__=='__main__':
    from time import clock, time, strftime
    from tqdm import tqdm
    from attention import *
    from keras.models import Model
    from keras.layers import *
    from keras.utils.np_utils import to_categorical
    from keras.optimizers import Adam
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('-epo', '--epochs', type=int, default=32)
    parser.add_argument('-bat', '--batch_size', type=int, default=128)
    parser.add_argument('-min', '--min_count', type=int, default=2)
    parser.add_argument('-emb', '--emb_size', type=int, default=512)
    parser.add_argument('-pos', '--pos_size', type=int, default=512)  ## concat
    parser.add_argument('-head', '--nb_head', type=int, default=8)
    parser.add_argument('-head_size', type=int, default=0)
    parser.add_argument('-resn', action='store_false')
    parser.add_argument('-bn', action='store_false')
    parser.add_argument('-ff', '--ff_size', type=int, nargs='?', const=0., default=2048)
    parser.add_argument('-div', type=float, default=10000.)
    parser.add_argument('-dropout', '--dropout_rate', type=float, default=0.5)
    parser.add_argument('-reg', type=float, default=2e-6)
    parser.add_argument('-bias_reg', type=float, default=0.)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-sub', action='store_false')
    parser.add_argument('-mask', '--layer_mask', action='store_false')
    parser.add_argument('-shuf', '--shuffle', action='store_false')
    parser.add_argument('-val', action='store_false')
    parser.add_argument('-tes', action='store_true')
    parser.add_argument('-rec', action='store_true')
    parser.add_argument('-quad', action='store_true')
    parser.add_argument('-nop', action='store_true')
    parser.add_argument('-embd', type=float, default=0.5)
    parser.add_argument('-embr', type=float, nargs='?', const=5e-8, default=0.)
    parser.add_argument('-smooth', type=float, nargs='?', const=0.1, default=0.)
    parser.add_argument('-es', type=int, nargs='?', const=0, default=5)
    parser.add_argument('--train_path', type=str, default='../icwb2/training')
    parser.add_argument('-cor', type=str, nargs='?', const='msr', default='pku')
    parser.add_argument('-train', '--train_file', type=str, default='')
    parser.add_argument('--test_path', type=str, default='../icwb2/testing')
    parser.add_argument('-test', '--test_file', type=str, default='')
    parser.add_argument('-gold', type=str, default='')
    parser.add_argument('-outp', type=str, default='ngrams_model_%s' % strftime('%y%m%d_%H%M%S'))  # str(int(100*time()))[-5:])
    parser.add_argument('-met', action='store_true')
    parser.add_argument('-univ', action='store_true')
    parser.add_argument('-concat', action='store_true')
    parser.add_argument('-res_char', action='store_true')
    parser.add_argument('-idp', action='store_true')
    # parser.add_argument('-neu', action='store_true')
    parser.add_argument('-avg', type=int, nargs='?', const=0, default=-1)
    parser.add_argument('-rev', action='store_false')
    parser.add_argument('-mo', action='store_false')
    parser.add_argument('-max_ngs', type=int, nargs='?', const=1, default=2)
    args = parser.parse_args()
    for arg in vars(args).keys():  # args.__dict__.keys()
        exec('%s = args.%s' % (arg, arg))
    size_per_head = head_size  if head_size else emb_size // nb_head
    po, bi = not quad, not nop; ngs_num = max_ngs - 1
    
    train_inf, test_inf = train_file if train_file else '%s_training.utf8' % cor, test_file if test_file else '%s_test.utf8' % cor
    output_file, score_file = os.path.join(outp, cor + '_test_epoch_%d.txt'), os.path.join(outp, 'score_epoch_%d.txt')
    if not os.path.isdir(outp): os.makedirs(outp); print('Model saved to %s' % outp)
    with open(os.path.join(outp, 'params.txt'), 'w') as f:
        json.dump({arg: eval('args.' + arg) for arg in args.__dict__.keys()}, f)

    X, Y, xlens, ngrams, char_idx, lb_idx, cnts = pre(train_path, train_inf, bi=bi, sub=sub, ngs_num=ngs_num, rev=rev,
                                                      smooth=smooth, min_count=min_count, batch_size=batch_size)

    nb_char, nb_class = len(char_idx) + ngs_num + 2, len(lb_idx) + 1
    print('-'*6 + '\nEmbedding Size: %d \n' % (nb_char))
    out_size, activ = (1, 'sigmoid') if bi else (nb_class, 'softmax')

    ngs_inputs, outputs = [], []
    inputs = Input(shape=(None, ), dtype='int32', name='char_seqs')
    mask_lens = Input(shape=(1, ), dtype='int32', name='mask_seqs')
    emb_layer = Embedding(nb_char, emb_size, embeddings_initializer='uniform', embeddings_regularizer=l2(embr))
    pos_layer = Position_Embedding(size=pos_size, mode='solo' if idp else 'sum', divide=div)
    char_emb = emb_layer(inputs)
    if idp: 
        pos_emb = pos_layer(char_emb)
    elif po: char_emb = pos_layer(char_emb)
    if embd: char_emb = Dropout(embd)(char_emb)

    att_layer = Attention(nb_head, size_per_head, mask_mode='mul', reg=2*reg)
    att_in = [pos_emb, char_emb, char_emb] if idp else [char_emb, char_emb, char_emb]
    if layer_mask: att_in += [mask_lens, mask_lens]
    x = att_layer(att_in)
    x = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)  # emb_size//2 if concat else emb_size
    if dropout_rate: x = Dropout(dropout_rate)(x)
    if concat: x = Concatenate()([x, char_emb])  # new_char = Dense(emb_size//2)(char_emb); 
    if resn and not concat: x = Add()([x, char_emb])
    if bn: x = BatchNormalization()(x)
    if ff_size: x = Dense(ff_size, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
    if mo: outputs.append(Dense(out_size, activation=activ, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg), name='o1')(x))
    
    # neurons = emb_size
    for n in range(ngs_num):
        # new_neurons = neurons + emb_size if neu else emb_size
        last_in = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
        ngs = Input(shape=(None, ), dtype='int32', name='%d_gram_seqs' % (n + 2))
        ngs_inputs.append(ngs)
        ngs_emb = emb_layer(ngs)
        if po and not idp: ngs_emb = pos_layer(ngs_emb)
        if embd: ngs_emb = Dropout(embd)(ngs_emb)
        att_in = [pos_emb, ngs_emb, last_in] if idp else [last_in, ngs_emb, last_in]
        if layer_mask: att_in += [mask_lens, mask_lens]
        x = att_layer(att_in) if univ else Attention(nb_head, size_per_head, mask_mode='mul', reg=2*reg)(att_in)
        x = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
        if dropout_rate: x = Dropout(dropout_rate)(x)
        if concat: x = Concatenate()([x, ngs_emb if res_char else last_in])
        if resn and not concat: x = Add()([x, ngs_emb if res_char else last_in])
        if bn: x = BatchNormalization()(x)
        if ff_size: x = Dense(ff_size, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
        if mo: outputs.append(Dense(out_size, activation=activ, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg), name='o%s' % (n + 2))(x))
        # neurons = new_neurons
    if not mo:
        if rec:
            x = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
            x = GRU(emb_size, return_sequences=True, implementation=2, dropout=dropout_rate, kernel_regularizer=l2(2*reg))(x)
            output = Dense(out_size, activation=activ, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg), name='out_rec')(x)
            outputs.append(output)
        else:
            output = Dense(out_size, activation=activ, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg), name='out_last')(x)
            outputs.append(output)


    # # O_seq = GRU(emb_size, return_sequences=True, implementation=2, dropout=dropout_rate)(O_seq)
    # # kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None
    
    model = Model(inputs=[inputs, mask_lens] + ngs_inputs, outputs=outputs)
    model.compile(loss='%s_crossentropy' % ('binary' if bi else 'categorical'),  # loss_weights=[1., 0.2]
                  optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-09),
                  metrics=['accuracy'])

    all_data, valids_size = [], len(X) // 6
    for i, (bat_x, bat_y, lens) in enumerate(zip(X, Y, xlens)):
        bat_x, lens, ngs = np.array(bat_x), np.array(lens), [np.array(ngrams[n][i]) for n in range(ngs_num)]
        bat_y = np.expand_dims(np.array(bat_y), axis=-1) if bi else to_categorical(bat_y, num_classes=nb_class).reshape(len(bat_y), -1, nb_class)
        all_data.append((bat_x, bat_y, lens, ngs))
    if not tes and val:
        trains, valids = all_data[valids_size:], all_data[:valids_size]
    else:
        trains = all_data
    X, xlens, ngrams, lines, sub_list, bat_idx = pre(test_path, test_inf, sub=sub, char_idx=char_idx, lb_idx=lb_idx, bi=bi, rev=rev,
                                                     ngs_num=ngs_num, test_set=True, batch_size=batch_size, max_seq_len=0, shuffle=False)
    start, sfl, loss, es_cnt, f1 = clock(), np.arange(len(trains)), 1e9, 0, 0
    disp_met = {'loss', 'acc'}
    print('-'*12 + '\nModel begins training...')
    for i in range(epochs):
        # valids = []
        print('\n' + '-'*6)
        outf, scf = output_file % i, score_file % i
        if shuffle: np.random.shuffle(sfl)
        total_len, avg_mtc = 0.0, np.array([0.0] * len(model.metrics_names))
        with tqdm(sfl) as batch:
            for j, k in enumerate(batch):
                bat_x, bat_y, lens, ngs = trains[k]
                # if j % 5 == 4:  # i % 5
                #     if not i: valids.append((bat_x, bat_y, lens))
                #     continue
                metrics = model.train_on_batch([bat_x, lens] + ngs, [bat_y for _ in range(int(not mo) or ngs_num+1)])
                batch.set_description('Train: Epoch %d' % i)
                batch.set_postfix(**{n: m for n, m in zip(model.metrics_names, metrics) if met or (n in disp_met)})
                avg_lens = np.average(lens)
                avg_mtc += np.array(metrics) * avg_lens; total_len += avg_lens
        avg_mtc /= total_len
        mtc = ['%s=%.4f' % (n, m) for n, m in zip(model.metrics_names, list(avg_mtc))]
        print('Epoch %d: Averaged metric on training set: %s' % (i, ', '.join(mtc)))
        model.save(os.path.join(outp, 'epoch_%d.h5' % i))
        if tes:
            inference()
            cmd = '../icwb2/scripts/score ../icwb2/gold/%s_training_words.utf8 %s %s > %s' % (cor, 
                  gold if gold else '../icwb2/gold/%s_test_gold.utf8' % cor, outf, scf)
            print('Scoring...')
            os.system(cmd)
            if not es: os.system('grep \'F MEASURE\' %s' % scf); continue
            with open(scf, 'rb') as f: obj = re.search('F MEASURE:\s+(0\.\d+)', f.read())
            if obj:
                print(obj.group())
                new_f1 = float(obj.group(1))
                if new_f1 <= f1 - 0.01:
                    es_cnt += 1
                    if es_cnt == es: break
                elif new_f1 > f1: f1 = new_f1
            continue
        if not val: continue
        # print('\nEpoch %d: Validating...' % i)
        total_len, avg_mtc = 0.0, np.array([0.0] * len(model.metrics_names))
        with tqdm(valids) as batch:
            for bat_x, bat_y, lens, ngs in batch:
                metrics = model.test_on_batch([bat_x, lens] + ngs, [bat_y for _ in range(int(not mo) or ngs_num+1)])
                batch.set_description('Validation: Epoch %d' % i)
                batch.set_postfix(**{n: m for n, m in zip(model.metrics_names, metrics) if met or (n in disp_met)})
                avg_lens = np.average(lens)
                avg_mtc += np.array(metrics) * avg_lens; total_len += avg_lens
        avg_mtc /= total_len
        mtc = ['%s=%.4f' % (n, m) for n, m in zip(model.metrics_names, list(avg_mtc))]
        print('Epoch %d: Averaged metric on validation set: %s' % (i, ', '.join(mtc)))
        if not es: continue
        cur_loss = avg_mtc[model.metrics_names.index('loss')]
        if cur_loss > loss + 0.01:
            print('Loss did not decrease on validation set, stopped after epoch %d. min loss: %.2f' % (i, loss))
            es_cnt += 1
            if es_cnt == es: break
        else: loss = cur_loss
    print('Model finishes training in %.2f seconds' % (clock() - start))
    if not tes:
        print('-'*12)
        cmd = '../icwb2/scripts/score ../icwb2/gold/%s_training_words.utf8 %s %s > %s; tail -n 14 %s' % (cor, 
              gold if gold else '../icwb2/gold/%s_test_gold.utf8' % cor, outf, scf, scf)
        inference()
        print('Scoring...')
        os.system(cmd)
        print('Score file saved to %s' % scf)
