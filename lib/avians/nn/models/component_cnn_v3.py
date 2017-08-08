from avians.nn.models import *

def train_model_from_dataset(dataset_dir,
                             feature_size=192,
                             samples_per_epoch=64,
                             batch_size=32,
                             nb_epoch=10000,
                             metrics=["accuracy"],
                             early_stop_patience=100,
                             validation_data=None,
                             validation_split=0.1):

    dataset_dir = os.path.expandvars(dataset_dir)
    class_dirs = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(dataset_dir + '/' + d)]
    n_classes = len(class_dirs)
    LOG.debug("=== n_classes ===")
    LOG.debug(n_classes)
    
    idg = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=40,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=1/255)
    
    classifier = create_model(img_rows=feature_size,
                              img_cols=feature_size,
                              img_channels=1,
                              nb_classes=n_classes,
                              metrics=metrics)

    # save_dir = '/tmp/generator-data-{}'.format(nr.randint(10000))
    # os.makedirs(save_dir)

    train_generator = idg.flow_from_directory(
        dataset_dir,
        target_size=(feature_size, feature_size),
        color_mode='grayscale',
        # save_to_dir=save_dir,
        # save_format='png',
        batch_size=samples_per_epoch)

    callbacks = []
    if early_stop_patience > 0: 
        stop_early = EarlyStopping(monitor='loss', 
                                   patience=early_stop_patience, 
                                   verbose=1, 
                                   mode='auto')
        callbacks += [stop_early]

    classifier.fit_generator(train_generator,
                             samples_per_epoch=samples_per_epoch,
                             nb_epoch=nb_epoch,
                             validation_data=train_generator,
                             nb_val_samples=samples_per_epoch/5,
                             callbacks=callbacks)
    return classifier
    
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
