from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def get_callbacks():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    return [early_stop, reduce_lr]
