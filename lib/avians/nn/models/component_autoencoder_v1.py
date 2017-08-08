
from avians.nn.models import * 

def train_model(X,
                batch_size=32,
                nb_epoch=100,
                metrics=["accuracy"],
                early_stop_patience=30,
                validation_data=None,
                validation_split=0.1):
    
    LOG.debug("=== X.shape ===")
    LOG.debug(X.shape)
    
    n_data, n_channels, n_rows, n_cols = X.shape
    
    # X = (X - 128) / 128

    X = X.astype('float32') / 255.

    autoencoder, decoder, encoder = create_model(img_rows=n_rows,
                                                 img_cols=n_cols,
                                                 img_channels=n_channels,
                                                 metrics=metrics)
                                                 
    autoencoder.summary()
    # np.random.shuffle(X)
    callbacks = []
    if early_stop_patience > 0: 
        stop_early = EarlyStopping(monitor='loss', 
                                   patience=early_stop_patience, 
                                   verbose=1, 
                                   mode='auto')
        callbacks += [stop_early]
    
    print("Fitting Autoencoder: {}".format(autoencoder))
    print("X: {}".format(X.shape))
    autoencoder.fit(X,
                    X,
                    batch_size=batch_size,
                    shuffle=True,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_data=(validation_data, validation_data) if np.any(validation_data) else None,
                    validation_split=validation_split,
                    callbacks=callbacks)
    return autoencoder, decoder, encoder

def create_model(img_rows=64,
                 img_cols=64, 
                 img_channels=1, 
                 # nb_filters=32, 
                 # nb_conv=2, 
                 # nb_pool=2, 
                 metrics=["accuracy"]):
    
    input_img = Input(shape=(img_channels, img_rows, img_cols))

    # input_img = Input(shape=(1, 28, 28))

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

    # x = Convolution2D(nb_filters, 3, 3, activation='relu', 
    #                   border_mode='same')(input_img)
    # x = MaxPooling2D((2, 2), border_mode='same')(x)
    # # x = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(x)
    # # x = MaxPooling2D((2, 2), border_mode='same')(x)
    # x = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(x)
    # encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    # # at this point the representation is (nb_filters, img_rows/8, img_cols/8)
    # x = Flatten()(x)
    # x = Dense(512)(x)
    # encoded = Dense(16)(x)

    # x = Dense(16)(encoded)
    # x = Dense(512)(x)
    # x = Reshape((8, 8, 8))(x)
    # x = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(x)
    # # x = UpSampling2D((2, 2))(x)
    # # x = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # decoded = Convolution2D(img_channels, 3, 3, activation='sigmoid', 
    #                         border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', 
                        loss='binary_crossentropy',
                        metrics=metrics)

    encoder = Model(input=input_img, output=encoded)
    encoder.compile(optimizer='adadelta', 
                    loss='binary_crossentropy',
                    metrics=metrics)

    encoded_shape = (8, img_rows//8, img_cols//8)
    encoded_input = Input(shape=encoded_shape)
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    decoder.compile(optimizer='adadelta', 
                    loss='binary_crossentropy',
                    metrics=metrics)

    return (autoencoder, decoder, encoder)
