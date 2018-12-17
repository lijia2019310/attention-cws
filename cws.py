# coding: utf-8
'''单层纯注意力分词模型'''
import os, re, keras, collections
import numpy as np

punc_cn = u'＂＃＄＆＇（）＊＋，：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿‘‛“”„‟…‧﹏﹑﹔！？｡。' # ·．–—％’／－
sub_strs = collections.OrderedDict()
sub_strs['nums'] = (u'[－\-＋+]?[0-9０-９]+(?:[.·．][0-9０-９]+)?[%％]?', '0')  # [－\-＋+’]?
sub_strs['nums_cn'] = (u'[零○一二三四五六七八九十百千万亿萬億]{2,}', '9')  # u'[壹贰叁肆伍陆柒捌玖拾佰仟]{2,}'
sub_strs['eng'] = (u'[Ａ-ＺA-Zａ-ｚa-z_.]+', 'a')

def pre(path, inf, encoding='utf8', sub=True, char_idx=None, lb_idx=None, bi=True,
        test_set=False, min_count=2, batch_size=32, max_seq_len=75, shuffle=True, smooth=0):
    with open(os.path.join(path, inf), 'rb') as f:
        corpus = f.read().decode(encoding).replace(u'\u3000', ' ')
    sub_list, lines, X, xlens, Y, two_grams = {'puncs': []}, [], [], [], [], []
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
        k, bat_x, lens, tgs = 0, [], [], []; i, j = bat_idx[k]; sent = lines[i][j]; max_len = len(sent)
        while len(sent) > 1:
            x, tg, last, pad = [], [], u'＾', [0]*(max_len - len(sent))
            for c in sent:
                x.append(char_idx.get(c, 1))        # 未登录字id为1
                tg.append(char_idx.get(last+c, 2))  # 未登录二元组为2
                last = c
            # bat_x.append([char_idx.get(c, 1) for c in lines[i][j]] + [0]*(max_len-len(lines[i][j])))
            bat_x.append(x + pad); tgs.append(tg + pad); lens.append(len(sent))
            k += 1; i, j = bat_idx[k]; sent = lines[i][j]
            if len(bat_x) == batch_size:
                X.append(bat_x); xlens.append(lens); two_grams.append(tgs)
                bat_x, lens, tgs, max_len = [], [], [], len(sent)
        if bat_x: X.append(bat_x); xlens.append(lens); two_grams.append(tgs)
        return X, xlens, two_grams, lines, sub_list, bat_idx

    labels, cnts = [], {}  # 第一次遍历训练语料，分行、分句并统计字频
    for sent in re.split('[%s\r\n]+' % punc_cn, corpus):
        if not sent.strip(): continue
        chars, lbs, last = u'', '', u'＾'
        for word in sent.split():
            lbs += 's' + 'a'*(len(word)-1) if bi else 's' if len(word) == 1 else 'b%se' % ('m'*(len(word)-2))
            chars += word
            for c in word: cnts[c] = cnts.get(c, 0) + 1; cnts[last+c] = cnts.get(last+c, 0) + 1; last = c
            if max_seq_len and len(chars) > max_seq_len:
                lines.append(chars)
                labels.append(lbs)
                chars, lbs = u'', ''
        lines.append(chars)
        labels.append(lbs)

    if not char_idx:
        i, char_idx = 3, {}   # cnts = sorted(cnts.items(), key=lambda x: x[1], reversed=True) # 0 for paddings, 1 for char, 2 for two_gram
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
        bat_x, bat_y, lens, tgs = [], [], [], []
        for j in bat_idx[st: end]:
            l = len(lines[j]); x, y, tg, last, pad = [], [], [], u'＾', [0] * (max_len - l)
            for k in range(l):
                c = lines[j][k]
                x.append(char_idx.get(c, 1))
                y.append(lb_idx[labels[j][k]])
                tg.append(char_idx.get(last+c, 2))
                last = c
            bat_x.append(x + pad); bat_y.append(y + pad); lens.append(l); tgs.append(tg + pad)
        X.append(bat_x); Y.append(bat_y); xlens.append(lens); two_grams.append(tgs)
        # X.append([[char_idx.get(c, 1) for c in lines[k]] + [0]*(max_len-len(lines[k])) for k in bat_idx[st: end]])
        # Y.append([[lb_idx[c] for c in labels[k]] + [0]*(max_len-len(labels[k])) for k in bat_idx[st: end]])
        # xlens.append([len(lines[k]) for k in bat_idx[st: end]])
    if shuffle:
        sfl = np.arange(len(X)); np.random.shuffle(sfl)
        X = [X[i] for i in sfl]; Y = [Y[i] for i in sfl]; xlens = [xlens[i] for i in sfl]; two_grams = [two_grams[i] for i in sfl]
    return X, Y, xlens, two_grams, char_idx, lb_idx, cnts

def inference():
    print('Model begins testing...')
    Y, sub_chars, sub_cnt = [], {sub_strs[strs][1]: strs for strs in sub_strs}, {sub_strs[strs][1]: 0 for strs in sub_strs}
    new_lines = [['' if len(sent) > 1 else sent for sent in line] for line in lines]
    if not bi: sep_pre, sep_post = {lb_idx[lb] for lb in 'bs'}, {lb_idx[lb] for lb in 'es'}; sep_pre.add(0)
    for bat_x, lens, tgs in tqdm(zip(X, xlens, two_grams)):
        bat_y = model.predict_on_batch([np.array(bat_x), np.array(lens), np.array(tgs)])
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
        new_lines[i] = line
            # ' ' + sub_list['puncs'][i].pop(0) + ' ' + lines[i].pop(0).strip() 
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
    from time import clock, time
    from tqdm import tqdm
    from attention import *
    from keras.models import Model
    from keras.layers import *
    from keras.utils.np_utils import to_categorical
    from keras.optimizers import Adam
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-epo', '--epochs', type=int, default=32)
    parser.add_argument('-bat', '--batch_size', type=int, default=128)
    parser.add_argument('-min', '--min_count', type=int, default=2)
    parser.add_argument('-emb', '--emb_size', type=int, default=512)
    parser.add_argument('-pos', '--pos_size', type=int, default=512)
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
    parser.add_argument('-outp', type=str, default='ngrams_model_%s' % str(int(100*time()))[-5:])
    args = parser.parse_args()
    for arg in vars(args).keys():  # args.__dict__.keys()
        exec('%s = args.%s' % (arg, arg))
    size_per_head = head_size  if head_size else emb_size // nb_head
    po, bi = not quad, not nop
    
    train_inf, test_inf = train_file if train_file else '%s_training.utf8' % cor, test_file if test_file else '%s_test.utf8' % cor
    output_file, score_file = os.path.join(outp, cor + '_test_epoch_%d.txt'), os.path.join(outp, 'score_epoch_%d.txt')
    if not os.path.isdir(outp): os.makedirs(outp); print('Model saved to %s' % outp)
    X, Y, xlens, two_grams, char_idx, lb_idx, cnts = pre(train_path, train_inf, bi=bi, sub=sub, smooth=smooth, 
                                                         min_count=min_count, batch_size=batch_size)
    nb_char, nb_class = len(char_idx) + 3, len(lb_idx) + 1  # 'sa' len(lb_idx)
    print('-'*6 + '\nEmbedding Size: %d \n' % (nb_char))
    out_size, activ = (1, 'sigmoid') if bi else (nb_class, 'softmax')

    inputs = Input(shape=(None, ), dtype='int32', name='char_seqs')
    mask_lens = Input(shape=(1, ), dtype='int32', name='mask_seqs')
    tgs = Input(shape=(None, ), dtype='int32', name='two_gram_seqs')
    emb_layer = Embedding(nb_char, emb_size, embeddings_initializer='uniform', embeddings_regularizer=l2(embr))   # mask_zero=True random_
    char_emb = emb_layer(inputs); tg_emb = emb_layer(tgs) 
    pos_layer = Position_Embedding(size=pos_size, mode='sum', divide=div)   # mode='solo'
    if po: char_emb = pos_layer(char_emb); tg_emb = pos_layer(tg_emb)
    if embd: tg_emb = Dropout(embd)(tg_emb); char_emb = Dropout(embd)(char_emb)

    # self_attention_1
    att_in_1 = [char_emb, tg_emb, char_emb, mask_lens, mask_lens] if layer_mask else [char_emb, tg_emb, char_emb]  # char_emb, tg_emb, char_emb
    x = Attention(nb_head, size_per_head, mask_mode='mul', reg=2*reg)(att_in_1)
    x = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)   # activation='relu'
    if dropout_rate: x = Dropout(dropout_rate)(x)           # noise_shape=(batch_size, 1, features)
    if resn: x = Add()([x, char_emb])
    if bn: x = BatchNormalization()(x)            # axis=-1, momentum=0.99, epsilon=0.001
    if ff_size: x = Dense(ff_size, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
    if rec:
        x = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
        out_1 = GRU(emb_size, return_sequences=True, implementation=2, dropout=dropout_rate, kernel_regularizer=l2(2*reg))(x)
        outputs = Dense(out_size, activation=activ, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg), name='output_labels')(out_1)
    else:
        outputs = Dense(out_size, activation=activ, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
    # self_attention_2
    # att_in_2 = [out_1, pos_emb, out_1, mask_lens, mask_lens] if layer_mask else [out_1, pos_emb, out_1]
    # x = Attention(nb_head, size_per_head, mask_mode='mul', reg=2*reg)(att_in_2)
    # x = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)
    # x = Dropout(dropout_rate)(x)
    # x = Add()([x, out_1])
    # att_out_2 = BatchNormalization()(x)
    # # x = Masking(mask_value=0.0)(att_out_2)
    # # out_2 = LSTM(out_size, return_sequences=True, implementation=2, dropout=dropout_rate,
    # #              kernel_regularizer=l2(2*reg), recurrent_regularizer=l2(2*reg))(x)
    # # outputs = Activation(activ)(out_2)
    # x = Dense(ff_size, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(att_out_2)
    # out_2 = Dense(emb_size, kernel_regularizer=l2(reg), bias_regularizer=l1(bias_reg))(x)

    # # self_attention_3
    # att_in_3 = [out_2, pos_emb, out_2, mask_lens, mask_lens] if layer_mask else [out_2, pos_emb, out_2]
    # out_3 = Attention(nb_head, size_per_head, mask_mode='add', reg=reg)(att_in_3)
    # # O_seq = Activation('relu')(O_seq)
    # # O_seq = GRU(emb_size, return_sequences=True, implementation=2, dropout=dropout_rate)(O_seq)
    # # kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None
    
    model = Model(inputs=[inputs, mask_lens, tgs], outputs=outputs)
    model.compile(loss='%s_crossentropy' % ('binary' if bi else 'categorical'),
                  optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-09),
                  metrics=['accuracy'])

    all_data, valids_size = [], len(X) // 6
    for bat_x, bat_y, lens, tgs in zip(X, Y, xlens, two_grams):
        bat_x, lens, tgs = np.array(bat_x), np.array(lens), np.array(tgs)
        bat_y = np.expand_dims(np.array(bat_y), axis=-1) if bi else to_categorical(bat_y, num_classes=nb_class).reshape(len(bat_y), -1, nb_class)
        all_data.append((bat_x, bat_y, lens, tgs))
    if not tes and val:
        trains, valids = all_data[valids_size:], all_data[:valids_size]
    else:
        trains = all_data
    X, xlens, two_grams, lines, sub_list, bat_idx = pre(test_path, test_inf, sub=sub, char_idx=char_idx, lb_idx=lb_idx, bi=bi,
                                                        test_set=True, batch_size=batch_size, max_seq_len=0, shuffle=False)
    start, sfl, loss, es_cnt, f1 = clock(), np.arange(len(trains)), 1e9, 0, 0
    print('-'*12 + '\nModel begins training...')
    for i in range(epochs):
        # valids = []
        print('-'*6)
        outf, scf = output_file % i, score_file % i
        if shuffle: np.random.shuffle(sfl)
        with tqdm(sfl) as batch:
            for j, k in enumerate(batch):
                bat_x, bat_y, lens, tgs = trains[k]
                # if j % 5 == 4:  # i % 5
                #     if not i: valids.append((bat_x, bat_y, lens))
                #     continue
                metrics = model.train_on_batch([bat_x, lens, tgs], bat_y)
                batch.set_description('Train: Epoch %d' % i)
                batch.set_postfix(**{n: m for n, m in zip(model.metrics_names, metrics)})
        model.save(os.path.join(outp, 'epoch_%d.h5' % i))
        if tes:
            inference()
            cmd = 'cd ../icwb2; scripts/score gold/pku_training_words.utf8 gold/pku_test_gold.utf8 ../src/%s > ../src/%s' % (outf, scf)
            print('Scoring...')
            os.system(cmd)
            if not es: os.system('grep \'F MEASURE\' ../src/%s' % scf); continue
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
            for bat_x, bat_y, lens, tgs in batch:
                metrics = model.test_on_batch([bat_x, lens, tgs], bat_y)
                batch.set_description('Validation: Epoch %d' % i)
                batch.set_postfix(**{n: m for n, m in zip(model.metrics_names, metrics)})
                avg_lens = np.average(lens)
                avg_mtc += np.array(metrics) * avg_lens; total_len += avg_lens
        avg_mtc /= total_len
        mtc = ['%s=%.2f' % (n, m) for n, m in zip(model.metrics_names, list(avg_mtc))]
        print('\nEpoch %d: Averaged metric on validation set: %s' % (i, ', '.join(mtc)))
        if not es: continue
        cur_loss = avg_mtc[model.metrics_names.index('loss')]
        if cur_loss > loss + 0.01:
            print('Loss did not decrease on validation set, stopped after epoch %d. min loss: %.2f' % (i, loss))
            es_cnt += 1
            if es_cnt == es: break
        else: loss = cur_loss
    print('\nModel finishes training in %.2f seconds' % (clock() - start))
    if not tes:
        print('-'*12)
        cmd = 'cd ../icwb2; scripts/score gold/pku_training_words.utf8 '\
              'gold/pku_test_gold.utf8 ../src/%s > ../src/%s; tail -n 14 ../src/%s' % (outf, scf, scf)
        inference()
        print('Scoring...')
        os.system(cmd)
    print('Score file saved to %s' % scf)
