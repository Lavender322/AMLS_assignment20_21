from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# CNN Classification Model
def model_B2(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    # Normalize the images
    training_images = (training_images/255)-0.5
    val_images = (val_images/255)-0.5
    test_images = (test_images/255)-0.5

    # Create a `Sequential` model
    model = Sequential([
        Conv2D(32, kernel_size=(5,5), strides=(1,1), # num_filter, filter_size
               activation='relu',
               input_shape=(20,30,3)),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Dropout(0.25),
        Conv2D(64, (5,5), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(1000, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax'),
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    callback = EarlyStopping(monitor='loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    clf = model.fit(training_images, training_labels,
                        batch_size=128,
                        epochs=100,
                        verbose=1,
                        validation_data=(val_images, val_labels),
                        callbacks=[callback])

    training_score = model.evaluate(training_images, training_labels, verbose=0)
    validation_score = model.evaluate(val_images, val_labels, verbose=0)
    test_score = model.evaluate(test_images, test_labels, verbose=0)
    print('Task B2')
    print(f"{'Number of epochs for training:':<31}{len(clf.history['loss']):>10.0f}")
    print(f"{'Training Set Accuracy Score:':<31}{training_score[1]:>10.4f}")
    print(f"{'Validation Set Accuracy Score:':<31}{validation_score[1]:>10.4f}")
    print(f"{'Test Set Accuracy Score:':<31}{test_score[1]:>10.4f}")

    return training_labels[1]