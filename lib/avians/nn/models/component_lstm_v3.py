from avians.nn.models import *
import avians.feature.projection as afp

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
        outdir = os.path.expandvars(os.path.join("$HOME/tmp/res-" + str(np.random.randint(1000)),
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
      
    classifier = Sequential()
    classifier.add(GRU(800,
                       return_sequences=True,
                       dropout_U=0.2,
                       dropout_W=0.2,
                       input_shape=(feature_height, longest_feature_len)))
    classifier.add(GRU(400,
                       dropout_U=0.1,
                       dropout_W=0.1,
                       return_sequences=True))
    classifier.add(GRU(400,
                       dropout_U=0.1,
                       dropout_W=0.1,
                       return_sequences=True))
    classifier.add(GRU(200,
                       dropout_U=0.1,
                       dropout_W=0.1,
                       return_sequences=False))
    classifier.add(Dense(8192, activation='relu'))
    classifier.add(Dense(n_labels+1, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', 
                       optimizer='rmsprop',
                       metrics=['accuracy'])

    classifier.summary()
    classifier.fit(X,
                   y,
                   batch_size=batch_size,
                   nb_epoch=nb_epoch,
                   callbacks=callbacks,
                   validation_split=validation_split)
    anc.save_and_upload_model(classifier,
                              outdir + '/',
                              "component_lstm_v3")
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


    # classifier.fit_generator(train_generator,
    #                          samples_per_epoch=samples_per_epoch,
    #                          nb_epoch=nb_epoch,
    #                          validation_data=train_generator,
    #                          nb_val_samples=samples_per_epoch/5,
    #                          callbacks=callbacks)
    # return classifier
