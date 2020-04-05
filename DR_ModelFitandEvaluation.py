#Below code trains model without using checkpoint
history_dropout = mdl.fit_generator(balanced_gen,
                                       steps_per_epoch = steps_per_epochs,
                                       epochs = N_epochs,
                                       validation_data = validation_generator,
                                       validation_steps = int(len(X_val)/N_batch_size)
                                       )

#Saing Model using Checkpoint approach**************************************************************************************************
#Create Checkpoint to save model after every epoch
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1,
    save_best_only=False, save_weights_only=False,
    save_frequency=1)

#Train model and save it using checkpoint
#Train is for N_epochs/2 
history_dropout = mdl.fit_generator(balanced_gen,callbacks=[checkpoint_callback],
                                       steps_per_epoch = steps_per_epochs,
                                       epochs = N_epochs/2, validation_freq=1,
                                       validation_data = validation_generator,
                                       validation_steps = int(len(X_val)/N_batch_size)
                                       )
#Train the saved model for the remaining N_epochs/2
history_dropout = mdl.fit_generator(balanced_gen,callbacks=[checkpoint_callback],
                                       steps_per_epoch = steps_per_epochs,
                                       epochs = N_epochs , validation_freq=1,
                                       validation_data = validation_generator,
                                       validation_steps = int(len(X_val)/N_batch_size), initial_epoch = N_epochs/2
                                       )

#Analyze the test loss**************************************************************************************************
results = mdl.evaluate(image_matrix_appended_test[0:N_test_examples],
                       label_matrix_appended_test[0:N_test_examples],
                       batch_size = N_batch_size)
print('test loss, test acc:', results)

#plot graphs for metrics accuracy and loss for training & validation datasets*******************************************
acc = history_dropout.history['accuracy']
val_acc = history_dropout.history['val_accuracy']

loss = history_dropout.history['loss']
val_loss = history_dropout.history['val_loss']

epochs_range = range(N_epochs)

plt.figure(figsize=(24, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Predict the target class for every test image**************************************************************************
for image in image_matrix_appended_test:
  image = image.reshape(1, N_image_size, N_image_size, 3)
  x = mdl.predict(image)
  print(x)
  target_class = np.argmax(x)
  print("Target Class = %d" %target_class)
  print('\n')
