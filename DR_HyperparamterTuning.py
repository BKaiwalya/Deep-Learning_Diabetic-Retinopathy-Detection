#*Hyperparameter Optimization starts************************************************************************************

from tensorflow.keras import layers

%load_ext tensorboard
!rm -rf ./logs/
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

#Model tuned for below values of hyperparameters:
#optimizer = SGD, RMSprop
#number of epochs = 100, 300
#number of neurons in the 2nd last dense layer = 128, 256, 512
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd','Adam']))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([100, 200]))
HP_DENSE_NEURONS = hp.HParam('dense_layer_neurons', hp.Discrete([128, 256, 512]))

#Choose the mertic as accuracy for hyperparameter tuning 
METRIC_ACCURACY = 'accuracy'

#Create logs for different hyperparameters
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_OPTIMIZER, HP_EPOCHS, HP_DENSE_NEURONS],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
#Function to write into the logs 
#Input: Log name, hyperparameter
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

#Define model with hyperparameters now
def train_test_model(hparams):

  inputs = keras.Input(shape=(N_image_size, N_image_size, 3))  # size of input initially 3
  x = layers.Conv2D(32, (3, 3), strides=2, activation='relu')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Conv2D(64, (3, 3), strides=2, activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPool2D((2, 2))(x)
  x = layers.Conv2D(128, (3, 3), strides=2, activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPool2D((2, 2))(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(hparams[HP_DENSE_NEURONS], activation='relu', kernel_regularizer=k.regularizers.l2(0.001))(x)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(2, activation='softmax')(x)

  mdl = keras.Model(inputs=inputs, outputs=outputs, name='DR_model')
  opt = hparams[HP_OPTIMIZER]
  mdl.compile(loss = k.losses.sparse_categorical_crossentropy, optimizer = opt, metrics=["accuracy"])

  #Train the model with hyperparameters
  mdl.fit_generator(balanced_gen,
                    steps_per_epoch = steps_per_epochs,
                    epochs = hparams[HP_EPOCHS],
                    validation_data = validation_generator,
                    validation_steps = int(len(X_val)/N_batchSize))
  _, accuracy = mdl.evaluate(image_matrix_appended_test[0:N_test_examples], label_matrix_appended_test[0:N_test_examples], batch_size = N_batchSize)
  return accuracy

#Create session. In every session logs are created and written into. 
session_num = 0
for dense_layer_neurons in HP_DENSE_NEURONS.domain.values:
  for optimizer in HP_OPTIMIZER.domain.values:
    for epochs in HP_EPOCHS.domain.values:
      hparams = {
          HP_OPTIMIZER: optimizer,
          HP_EPOCHS: epochs,
          HP_DENSE_NEURONS: dense_layer_neurons
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1
      
#Open the logs in tensorboard
%tensorboard --logdir logs/hparam_tuning

##*Hyperparameter Optimization ends**************************************************************************************
