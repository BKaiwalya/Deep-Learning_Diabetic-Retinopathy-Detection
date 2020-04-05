#Functional API for the model

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
x = layers.Dense(512, activation='relu', kernel_regularizer=k.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation='softmax')(x)

mdl = keras.Model(inputs=inputs, outputs=outputs, name='DR_model')

#Define Optimizer and Loss function
opt = tf.optimizers.SGD(learning_rate)
mdl.compile(loss = k.losses.sparse_categorical_crossentropy, optimizer = opt, metrics=["accuracy"])

#Model written using MyModel as a class 
"""
class MyModel(k.Model):
  def __init__(self):
  # Create layers
    super(MyModel, self).__init__()
    self.conv0 = k.layers.Conv2D(32, (3 ,3), activation='relu',  input_shape = (N_image_size, N_image_size, 1))
    self.batch0 = tf.keras.layers.BatchNormalization()
    self.pool0 = tf.keras.layers.MaxPool2D((2, 2))
    self.conv1 = k.layers.Conv2D(32, (3 ,3), activation='relu')
    self.batch1 = tf.keras.layers.BatchNormalization()
    self.pool1 = tf.keras.layers.MaxPool2D((2, 2))
    self.conv2 = k.layers.Conv2D(64, (3 ,3), activation='relu')
    self.batch2 = tf.keras.layers.BatchNormalization()
    self.pool2 = tf.keras.layers.MaxPool2D((2, 2))
    self.conv3 = k.layers.Conv2D(64, (3 ,3), activation='relu')
    self.batch3 = tf.keras.layers.BatchNormalization()
    self.pool3 = tf.keras.layers.MaxPool2D((2, 2))
    self.conv4 = tf.keras.layers.Conv2D(128, (3 ,3), activation='relu')
    self.batch4 = tf.keras.layers.BatchNormalization()
    self.pool4 = tf.keras.layers.MaxPool2D((2, 2))
    self.conv5 = tf.keras.layers.Conv2D(128, (3 ,3), activation='relu')
    #self.batch5 = tf.keras.layers.BatchNormalization()
    self.pool5 = tf.keras.layers.MaxPool2D((2, 2))
    #self.conv6 = tf.keras.layers.Conv2D(256, (3 ,3), activation='relu')
    #self.conv7 = tf.keras.layers.Conv2D(256, (3 ,3), activation='relu')
    #self.pool6 = tf.keras.layers.MaxPool2D((2, 2))
    self.dropout0 = k.layers.Dropout(0.5)
    self.flatten0 = k.layers.Flatten()
    self.dense0 = k.layers.Dense(512,activation='relu')#,kernel_regularizer=k.regularizers.l2(0.0001))
    self.dense1 = k.layers.Dense(128,activation='relu')#,kernel_regularizer=k.regularizers.l2(0.001))
    self.dropout1 = k.layers.Dropout(0.5)
    self.dense3 = k.layers.Dense(2,activation='softmax')

  def call(self, inputs, training=False):
    # Implement forward pass
    #output = self.flatten0(inputs)
    output = self.conv0(inputs)
    output = self.batch0(output)
    output = self.pool0(output)
    output = self.conv1(output)
    output = self.batch1(output)
    output = self.pool1(output)
    output = self.conv2(output)
    output = self.batch2(output)
    output = self.pool2(output)
    output = self.conv3(output)
    output = self.batch3(output)
    output = self.pool3(output)
    output = self.conv4(output)
    output = self.batch4(output)
    output = self.pool4(output)
    output = self.conv5(output)
    output = self.pool5(output)
    output = self.dropout0(output)
    output = self.flatten0(output)
    output = self.dense0(output)
    output = self.dense1(output)
    output = self.dropout1(output)
    output = self.dense3(output)
    return output
"""
