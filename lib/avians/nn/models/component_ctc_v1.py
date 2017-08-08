from avians.nn.models import *
import avians.feature.projection as afp


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def train_model(dataset_dir,
                feature_size=192,
                samples_per_epoch=64,
                batch_size=512,
                nb_epoch=1000,
                metrics=["accuracy"],
                early_stop_patience=30,
                validation_data=None,
                validation_split=0.1,
                outdir=None):

    if not outdir: 
        outdir = os.path.expandvars(
            os.path.join("$HOME/tmp/res-" + str(np.random.randint(1000)),
                         os.path.basename(dataset_dir)))

    if not os.path.isdir(outdir): 
        os.makedirs(outdir)

    dataset_dir = os.path.expandvars(dataset_dir)
    class_dirs = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(dataset_dir + '/' + d)]
    n_classes = len(class_dirs)
    
    samples = []
    longest_label_len = 0
    longest_feature_len = 0
    label_set = set()
    feature_height = 64
    
    for cd in class_dirs:
        # LOG.debug("=== class_dirs ===")
        # LOG.debug(class_dirs)
        image_files = aui.file_list(os.path.join(dataset_dir, cd))
        label = cd
        label_set.add(label)
        if len(label) > longest_label_len: 
            longest_label_len = len(label)
        for imgf in image_files:
            # LOG.debug("=== cd ===")
            # LOG.debug(cd)
            
            # LOG.debug("=== imgf ===")
            # LOG.debug(imgf)
            
            img = cv2.imread(os.path.join(cd, imgf), 0)
            imgr = cv2.resize(img, dsize=(img.shape[1], feature_height))
            imgf = np.zeros((imgr.shape[0], imgr.shape[1]), np.bool)
            imgf[imgr.nonzero()] = 1
            longest_feature_len = (imgf.shape[1] 
                                   if (imgf.shape[1] > longest_feature_len) 
                                   else longest_feature_len)
            samples.append((label, imgf))

    label_num_dict = {l: i for i, l in enumerate(label_set)}
    num_label_dict = {i: l for l, i in label_num_dict.items()}
    n_labels = len(label_set)

    X = np.zeros(shape=(len(samples),
                        feature_height,
                        longest_feature_len),
                 dtype=np.bool)

    y = np.zeros(shape=(len(samples),
                        n_labels + 1),
                 dtype=np.bool)

    LOG.debug("=== X.shape ===")
    LOG.debug(X.shape)

    for i, s in enumerate(samples):
        lab, feature = s
        LOG.debug("=== feature.shape ===")
        LOG.debug(feature.shape)

        X[i, :, :feature.shape[1]] = feature
        label_i = label_num_dict[lab]

        y[i, label_i] = 1

    class_table = np.zeros(shape=(len(label_num_dict),),
                           dtype=[("label", "S32"),
                                  ("num", np.int)])
    
    for i, l in enumerate(label_num_dict): 
        n = label_num_dict[l]
        class_table[i]['label'] = l
        class_table[i]['num'] = n

    dataset_outfile = outdir + "/component_lstm_v3.npz"
    np.savez_compressed(dataset_outfile, 
                        X=X,
                        y=y,
                        class_table=class_table)

    anc.shuffle_parallel(X, y)

    LOG.debug("=== X ===")
    LOG.debug(X)
    
    LOG.debug("=== y.shape ===")
    LOG.debug(y.shape)

    LOG.debug("=== y ===")
    LOG.debug(y)

    callbacks = []
    if early_stop_patience > 0: 
        stop_early = EarlyStopping(monitor='loss', 
                                   patience=early_stop_patience, 
                                   verbose=1, 
                                   mode='auto')
        callbacks += [stop_early]

    n_conv_f = 16
    s_conv_f = 3
    n_pool_1 = 4
    n_pool_2 = 2
      
    the_input = Input(shape=(1, 
                             feature_height, 
                             longest_feature_len),
                      dtype='float32')
    classifier = Convolution2D(n_conv_f,
                               s_conv_f, 
                               s_conv_f,
                               border_mode='same',
                               activation="relu")(
                                   the_input)
    classifier = MaxPooling2D(pool_size=(n_pool_1,
                                         n_pool_1))(
                                             classifier)
    classifier = Convolution2D(n_conv_f,
                               s_conv_f, 
                               s_conv_f,
                               border_mode='same',
                               activation="relu")(
                                   classifier)
    classifier = MaxPooling2D(pool_size=(n_pool_2,
                                         n_pool_2))(
                                             classifier)

    reshape_dims = (int((feature_height / (n_pool_1 * 
                                           n_pool_2)) * 
                        n_conv_f), 
                    int(longest_feature_len / (n_pool_1 *
                                               n_pool_2)))

    classifier = Reshape(target_shape=reshape_dims)(classifier)
    classifier = Permute(dims=(2, 1))(classifier)

    classifier = TimeDistributed(Dense(32, 
                                       activation='relu'))(
                                           classifier)

    gru_1 = GRU(512,
                return_sequences=True)(classifier)
    gru_1b = GRU(512,
                 go_backwards=True, 
                 return_sequences=True)(classifier)
    classifier = merge([gru_1, gru_1b], mode='sum')
    gru_2 = GRU(512,
                 return_sequences=True)(classifier)
    gru_2b = GRU(512,
                 go_backwards=True, 
                 return_sequences=True)(classifier)
    classifier = merge([gru_2, gru_2b], mode='concat')

    classifier = TimeDistributed(Dense(longest_label_len))(
        classifier)

    classifier = Activation('softmax')(classifier)
    
    Model(input=[the_input], output=classifier).summary()

    labels = Input(name='the_labels',
                   shape=[longest_label_len],
                   dtype='float32')
    input_length = Input(name='input_length', 
                         shape=[1], 
                         dtype='int64')
    label_length = Input(name='label_length', 
                         shape=[1], 
                         dtype='int64')

    loss_out = Lambda(ctc_lambda_func, 
                      output_shape=(1,),
                      name="ctc")([classifier, 
                                   labels, 
                                   input_length,
                                   label_length])

    sgd = SGD(lr=0.03, 
              decay=3e-7, 
              momentum=0.9, 
              nesterov=True, 
              clipnorm=5)

    model = Model(input=[the_input, 
                         labels, 
                         input_length, 
                         label_length], 
                  output=[loss_out])

    model.compile(loss={'ctc': 
                        lambda y_true, y_pred: y_pred}, 
                  optimizer=sgd)

    model.fit(X,
              y,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              callbacks=callbacks,
              validation_split=validation_split)

    anc.save_and_upload_model(model,
                              outdir + '/',
                              "component_ctc_v1")
    return model

