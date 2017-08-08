from avians.nn.models import *
import avians.feature.chain_code as afcc

def prepare_dataset(dataset_dir): 
    dataset_dir = os.path.expandvars(dataset_dir)
    class_dirs = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(dataset_dir + '/' + d)]
    n_classes = len(class_dirs)
    
    samples = []
    longest_label_len = 0
    longest_cc_len = 0
    label_set = set()
    
    for cd in class_dirs:
        # LOG.debug("=== class_dirs ===")
        # LOG.debug(class_dirs)
        image_files = aui.file_list(os.path.join(dataset_dir, cd))
        label = cd
        for c in list(label): 
            label_set.add(c)
        if len(label) > longest_label_len: 
            longest_label_len = len(label)
        for imgf in image_files:
            # LOG.debug("=== cd ===")
            # LOG.debug(cd)
            
            # LOG.debug("=== imgf ===")
            # LOG.debug(imgf)
            
            img = cv2.imread(os.path.join(cd, imgf), 0)
            cc = afcc.chain_code_from_img(img)
            if len(cc) > longest_cc_len: 
                longest_cc_len = len(cc)
            samples.append((label, cc))

    
    num_label_chars = len(label_set)
    char_list = [(ch, i + 1) for i, ch in enumerate(label_set)]
    char_index_dict = {ch: i for ch, i in char_list}
    index_char_dict = {i: ch for ch, i in char_list}

    X = np.zeros(shape=(len(samples),
                        longest_cc_len,
                        9),
                 dtype=np.bool)

    y = np.zeros(shape=(len(samples),
                        longest_label_len + 1),
                 dtype=np.bool)
    
    for i, s in enumerate(samples):
        lab, cc = s
        # LOG.debug("=== cc ===")
        # LOG.debug(cc)
        
        for j, c in enumerate(cc):
            # LOG.debug("=== c ===")
            # LOG.debug(c)
            
            X[i, j, c - 1] = 1
        # LOG.debug("=== X[i] ===")
        # LOG.debug(X[i])
        
        y[i, len(lab)] = 1
        # LOG.debug("=== y[i] ===")
        # LOG.debug(y[i])

    return X, y, longest_cc_len, longest_label_len

def train_model(dataset_dir,
                feature_size=192,
                samples_per_epoch=64,
                batch_size=32,
                nb_epoch=10000,
                metrics=["accuracy"],
                early_stop_patience=100,
                validation_data=None,
                validation_split=0.1):

    outdir = "/tmp/" + os.path.basename(dataset_dir)
    outfile = outdir + "/component_lstm_v1.npz"
    if os.path.exists(outfile):
        print("Loading from: {}".format(outfile))
        ds = np.load(outfile)
        X = ds['X']
        y = ds['y']
    else:
        print("Preparing Dataset")
        X, y = prepare_dataset(dataset_dir)
        if not os.path.exists(outdir): 
            os.makedirs(outdir)
        np.savez_compressed(outfile, 
                            X=X,
                            y=y)
        print("Saving to: {}".format(outfile))

    
    anc.shuffle_parallel(X, y)

    LOG.debug("=== X.shape ===")
    LOG.debug(X.shape)
    

    LOG.debug("=== X ===")
    LOG.debug(X)
    
    LOG.debug("=== y.shape ===")
    LOG.debug(y.shape)
    

    LOG.debug("=== y ===")
    LOG.debug(y)
    

    classifier = Sequential()
    classifier.add(LSTM(256,
                        return_sequences=False,
                        stateful=True,
                        input_shape=(longest_cc_len, 9)))
    classifier.add(Dense(4096, activation='tanh'))
    classifier.add(Dense(longest_label_len + 1, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', 
                       optimizer='rmsprop',
                       metrics=['accuracy'])

    classifier.summary()
    classifier.fit(X,
                   y,
                   batch_size=128,
                   nb_epoch=10,
                   validation_split=0.1)
    return classifier

    # # save_dir = '/tmp/generator-data-{}'.format(nr.randint(10000))
    # # os.makedirs(save_dir)

    # train_generator = idg.flow_from_directory(
    #     dataset_dir,
    #     target_size=(feature_size, feature_size),
    #     color_mode='grayscale',
    #     # save_to_dir=save_dir,
    #     # save_format='png',
    #     batch_size=samples_per_epoch)

    # callbacks = []
    # if early_stop_patience > 0: 
    #     stop_early = EarlyStopping(monitor='loss', 
    #                                patience=early_stop_patience, 
    #                                verbose=1, 
    #                                mode='auto')
    #     callbacks += [stop_early]

    # classifier.fit_generator(train_generator,
    #                          samples_per_epoch=samples_per_epoch,
    #                          nb_epoch=nb_epoch,
    #                          validation_data=train_generator,
    #                          nb_val_samples=samples_per_epoch/5,
    #                          callbacks=callbacks)
    # return classifier
    
def create_model(img_rows, 
                 img_cols, 
                 img_channels,
                 nb_classes, 
                 metrics=['accuracy']):

    input_img = Input(shape=(img_channels, img_rows, img_cols))
    x = Convolution2D(32, 3, 3, activation='relu', 
                      border_mode='same')(input_img)
    x = Convolution2D(32, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', 
                      border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # x = Convolution2D(512, 3, 3, activation='relu', 
    #                   border_mode='same')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', 
    #                   border_mode='same')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', 
    #                   border_mode='same')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    # x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(4096, activation="relu")(x)
    classifier = Dense(nb_classes, activation="softmax")(x)

    classifier_model = Model(input_img, classifier)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    classifier_model.compile(loss='categorical_crossentropy', 
                             optimizer=sgd,
                             metrics=metrics)

    return classifier_model
