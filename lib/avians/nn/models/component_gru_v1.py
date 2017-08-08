from avians.nn.models import *
import avians.feature.projection as afp

def prepare_dataset(dataset_dir):
    dataset_dir = os.path.expandvars(dataset_dir)
    class_dirs = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(dataset_dir + '/' + d)]
    n_classes = len(class_dirs)
    
    samples = []
    longest_label_len = 0
    longest_feature_len = 0
    label_set = set()
    feature_height = 48
    feature_width = 512

    char_num_dict = {}
    num_char_dict = {}
    char_num_index = 1
    
    for cd in class_dirs:
        # LOG.debug("=== class_dirs ===")
        # LOG.debug(class_dirs)
        image_files = aui.file_list(os.path.join(dataset_dir, cd))
        label = cd
        for imgf in image_files:
            # LOG.debug("=== cd ===")
            # LOG.debug(cd)
            
            # LOG.debug("=== imgf ===")
            # LOG.debug(imgf)
            
            img = cv2.imread(os.path.join(cd, imgf), 0)
            # convert image to a projection with 
            imgr = cv2.resize(img, dsize=(feature_width, feature_height), 
                              interpolation=cv2.INTER_AREA)
            imgf = afp.projection_from_img(imgr)
            # LOG.debug("=== imgf.shape before ===")
            # LOG.debug(imgf.shape)
            
            # imgf = imgf[0] + imgf[1] * feature_height + imgf[2] * feature_height * feature_height
            # LOG.debug("=== imgf.shape merged ===")
            # LOG.debug(imgf.shape)
            
            longest_feature_len = (imgf.shape[1]
                                   if (imgf.shape[1] > longest_feature_len) 
                                   else longest_feature_len)
            # for l in label:
            if label not in char_num_dict:
                char_num_dict[label] = char_num_index
                num_char_dict[char_num_index] = label
                char_num_index += 1
            # Reverse because Arabic
            # numeric_lab = np.array([char_num_dict[l] for l in label[::-1]])
            samples.append((char_num_dict[label], imgf))

    # longest_label_len = max([len(nl) for nl, imgf in samples])
    X = np.zeros(shape=(len(samples),
                        longest_feature_len * 3),
                 dtype=np.int)
    y = np.zeros(shape=(len(samples),
                        char_num_index),
                 dtype=np.bool)
                              
    LOG.debug("=== X.shape ===")
    LOG.debug(X.shape)
    LOG.debug("=== longest_label_len ===")
    LOG.debug(longest_label_len)
    LOG.debug("=== y.shape ===")
    LOG.debug(y.shape)
    w = longest_feature_len
    for i, s in enumerate(samples):
        lab, feature = s
        # LOG.debug("=== feature.shape ===")
        # LOG.debug(feature.shape)
        X[i, :w] = feature[0]
        X[i, w:(w*2)] = feature[1]
        X[i, (w*2):(w*3)] = feature[2]
        y[i, lab] = 1

    label_num_table = np.zeros(shape=(char_num_index,),
                               dtype=[("label", "S32"),
                                      ("num", np.int)])
    
    for i, l in enumerate(char_num_dict): 
        n = char_num_dict[l]
        label_num_table[i]['label'] = l
        label_num_table[i]['num'] = n

    return X, y, label_num_table


def train_model(dataset_dir,
                batch_size=32,
                nb_epoch=10000,
                metrics=["accuracy"],
                early_stop_patience=100,
                validation_data=None,
                validation_split=0.1):

    outdir = "/tmp/" + os.path.basename(dataset_dir)
    outfile = outdir + "/component_gru_v1.npz"
    if os.path.exists(outfile):
        print("Loading from: {}".format(outfile))
        ds = np.load(outfile)
        X = ds['X']
        y = ds['y']
    else:
        print("Preparing Dataset")
        X, y, class_table = prepare_dataset(dataset_dir)
        os.makedirs(outdir)
        np.savez_compressed(outfile, 
                            X=X,
                            y=y,
                            class_table=class_table)
        print("Saving to: {}".format(outfile))

    anc.shuffle_parallel(X, y)

    # LOG.debug("=== X ===")
    # LOG.debug(X)
    
    # LOG.debug("=== y ===")
    # LOG.debug(y)

    LOG.debug("=== X.max() ===")
    LOG.debug(X.max())
    
    embedding_input_size = X.shape[1]
    LOG.debug("=== embedding_input_size ===")
    LOG.debug(embedding_input_size)
    embedding_vector_size = 48
    
    classifier = Sequential()
    classifier.add(Embedding(embedding_input_size, 
                             embedding_vector_size,
                             input_length = X.shape[1]))
    classifier.add(Convolution1D(nb_filter=48, filter_length=5, border_mode='same', activation='relu'))
    classifier.add(MaxPooling1D(pool_length=2))
    classifier.add(Convolution1D(nb_filter=48, filter_length=7, border_mode='same', activation='relu'))
    classifier.add(MaxPooling1D(pool_length=2))
#    classifier.add(Dropout(0.1))
    classifier.add(Convolution1D(nb_filter=48, filter_length=9, border_mode='same', activation='relu'))
    classifier.add(MaxPooling1D(pool_length=2))
    classifier.add(GRU(128))
#    classifier.add(Dropout(0.1))
    # classifier.add(Convolution1D(nb_filter=64, filter_length=5, border_mode='same', activation='relu'))
    # classifier.add(Convolution1D(nb_filter=64, filter_length=5, border_mode='same', activation='relu'))
    # classifier.add(MaxPooling1D(pool_length=2))
#   classifier.add(Flatten())
    # classifier.add(Dense(4000, activation='relu'))
    # classifier.add(Dense(2000, activation='relu'))
    # classifier.add(Dense(1000, activation='relu'))
    # classifier.add(GRU(128,
    #                    return_sequences=False,
    #                    activation='tanh',
    #                    dropout_U=0.1,
    #                    dropout_W=0.1))
    # classifier.add(GRU(256,
    #                    return_sequences=True,
    #                    activation='tanh',
    #                    dropout_U=0.1,
    #                    dropout_W=0.1))
    # classifier.add(GRU(256,
    #                    return_sequences=True,
    #                    activation='tanh',
    #                    dropout_U=0.1,
    #                    dropout_W=0.1))
    # classifier.add(GRU(64,
    #                    return_sequences=False,
    #                    activation='tanh',
    #                    dropout_U=0.1,
    #                    dropout_W=0.1))
    # classifier.add(GRU(longest_label_len+1,
    #                    return_sequences=False,
    #                    activation='relu',
    #                    dropout_U=0.1,
    #                    dropout_W=0.1))
    classifier.add(Dense(y.shape[1],
                         activation='softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    classifier.compile(loss='categorical_crossentropy', 
                       optimizer='rmsprop',
                       metrics=['accuracy'])
    classifier.summary()
    classifier.fit(X,
                   y,
                   batch_size=batch_size,
                   nb_epoch=100,
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
