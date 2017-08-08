
from avians.nn.models import * 

def train_model(X, 
                y,
                batch_size=32,
                nb_epoch=100,
                metrics=["accuracy"],
                early_stop_patience=30,
                validation_split=0.1): 
    
    n_data, n_rows, n_cols, n_channels = X.shape
    X = X.swapaxes(2, 3).swapaxes(1, 2)
    X = X.astype(np.float) / 255
    n_classes = len(np.unique(y)) + 1
    y = np_utils.to_categorical(y, n_classes)
    model = create_model(img_rows=n_rows,
                         img_cols=n_cols,
                         img_channels=n_channels,
                         nb_filters=n_rows//2,
                         nb_classes=n_classes,
                         metrics=metrics)
    model.summary()
    anc.shuffle_parallel(X, y)
    callbacks = []
    if early_stop_patience > 0: 
        stop_early = EarlyStopping(monitor='loss', 
                                   patience=early_stop_patience, 
                                   verbose=1, 
                                   mode='auto')
        callbacks += [stop_early]
    
    print("X: {}".format(X.shape))
    print("y: {}".format(y.shape))
    model.fit(X,
              y, 
              batch_size=batch_size,
              shuffle=True,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_split=validation_split,
              callbacks=callbacks)
    return model

def create_model(img_rows, 
                 img_cols, 
                 img_channels, 
                 nb_classes=64,
                 nb_filters=32,
                 nb_conv=2,
                 nb_pool=2,
                 nb_dense_1=512,
                 nb_dense_2=1024,
                 metrics=['accuracy']):

    input_img = Input(shape=(img_channels, img_rows, img_cols))
    x = Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu', 
                      border_mode='same')(input_img)
    x = Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu', 
                      border_mode='same')(x)
    x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)
    x = Dropout(0.1)(x)

    x = Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu', 
                      border_mode='same')(x)
    x = Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu', 
                      border_mode='same')(x)
    x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)
    x = Dropout(0.1)(x)

    x = Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu', 
                      border_mode='same')(x)
    x = Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu', 
                      border_mode='same')(x)
    x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)
    x = Dropout(0.1)(x)

    x = Flatten()(x)
    x = Dense(nb_dense_1, activation="relu")(x)
    x = Dense(nb_dense_2, activation="relu")(x)
    x = Dense(nb_classes, activation="softmax")(x)

    model = Model(input_img, x)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='rmsprop', 
                  metrics=metrics)
 
    return model


