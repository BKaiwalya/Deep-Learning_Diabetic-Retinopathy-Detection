#Guided BackPropagation with GradCAM*********************************************************************

import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow

#Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([mdl.input], [mdl.get_layer('conv2d_1').output, mdl.output])

#Predict the label and get true label for all test samples
for i in range(0, N_test_examples):
  label_ground_truth = label_matrix_appended_test[i]
  image_ground_truth = image_matrix_appended_test[i]
  image_ground_truth = image_ground_truth.reshape(1, 256, 256, 3) 
  prediction = mdl.predict(image_ground_truth)
  #print(prediction)
  target_class = np.argmax(prediction)
  #print("Target Class = %d" %target_class)

  #Compute the loss between predicted label and true label for particular image 
  with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(image_ground_truth)
      loss = predictions[:, target_class]
  
  #Compute gradient of loss obtained with respect to the output of the last convolution layer
  #Extract filter and gradients
  output = conv_outputs[0]
  grads = tape.gradient(loss,conv_outputs)[0]

  #Take only those gradients and outputs which are positive. 
  #Zero out the negative gradients or gradients associated with negative value of filter.
  #This is to eliminate elements that act negatively towards decision
  gate_f = tf.cast(output > 0, 'float32')
  gate_r = tf.cast(grads > 0, 'float32')
  guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

  #Average the gradients and guided gradients spatially
  weights = tf.reduce_mean(guided_grads, axis=(0, 1))
  weights_gradcam = tf.reduce_mean(grads, axis=(0, 1))

  #Build a map of filters according to importance of gradients. 
  cam = np.ones(output.shape[0:2], dtype=np.float32)
  cam_gradcam = np.ones(output.shape[0:2], dtype=np.float32)

  for index, w in enumerate(weights):
    cam += w * output[:,:,index]

  for index, w in enumerate(weights_gradcam):
    cam_gradcam += w * output[:,:,index]

  #Heatmap Visualization
  cam = cv2.resize(cam.numpy(), (256, 256))
  cam_gradcam = cv2.resize(cam_gradcam.numpy(), (256, 256))

  cam = np.maximum(cam, 0)
  cam_gradcam = np.maximum(cam_gradcam, 0)

  heatmap = (cam - cam.min()) / (cam.max() - cam.min())
  heatmap_gradcam = (cam_gradcam - cam_gradcam.min()) / (cam_gradcam.max() - cam_gradcam.min())

  image_ground_truth = image_ground_truth.reshape(256,256,3)

  cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
  output_image = cv2.addWeighted(cv2.cvtColor(image_ground_truth.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
  
  cam_gradcam = cv2.applyColorMap(np.uint8(255*heatmap_gradcam), cv2.COLORMAP_JET)
  output_image_gradcam = cv2.addWeighted(cv2.cvtColor(image_ground_truth.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam_gradcam, 1.0, 0)
  
  #Saving images from google colab to drive
  path = '/content/drive/My Drive/GradCam_Images/'+ 'GuidedGrad-CAM_' +  str(i) + '.png'

  if (label_ground_truth == 0):
    label_ground_truth = 'NRDR'
  else:
    label_ground_truth = 'RDR'

  if (target_class == 0):
    target_class = 'NRDR'
  else:
    target_class = 'RDR'
  
  b,g,r = cv2.split(output_image)       
  output_image = cv2.merge([r,g,b])
    
  b,g,r = cv2.split(output_image_gradcam)       
  output_image_gradcam = cv2.merge([r,g,b])  

  plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.xlabel('Label Ground Truth:'+ ' ' + str(label_ground_truth))
  plt.imshow(image_ground_truth)
  plt.title('Image Ground Truth')
  
  plt.subplot(1, 3, 2)
  plt.imshow(output_image_gradcam)
  plt.xlabel('Label Predicted:'+  ' ' + str(target_class))
  plt.title('GradCAM')

  plt.subplot(1, 3, 3)
  plt.imshow(output_image)
  plt.title('GradCAM + GuidedBackProp')
  plt.xlabel('Label Predicted:'+  ' ' + str(target_class))
  plt.savefig(path, bbox_inches='tight')
  
  
 #Visualize Kernel***********************************************************************************
#Kernel Visualization

import numpy as np
import tensorflow as tf

# Layer name to inspect
layer_name = 'conv2d_1'

epochs = 50
step_size = 1.
filter_index = 0

# Create a connection between the input and the target layer
model = mdl
submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])

# Initiate random noise
input_img_data = image_matrix_appended_test[1].reshape(1, 256, 256, 3)
input_img_data = (input_img_data - 0.5) * 20 + 128.

input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

# Iterate gradient ascents
for _ in range(epochs):
    with tf.GradientTape() as tape:
        outputs = submodel(input_img_data)
        loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
    grads = tape.gradient(loss_value, input_img_data)
    normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    input_img_data.assign_add(normalized_grads * step_size)

input_img_data = tf.Variable(tf.cast(input_img_data, tf.int32))


input_img_data = tf.reshape(input_img_data, [256, 256, 3])
print(input_img_data.shape)

input_img_data = np.asarray(input_img_data) 
print(input_img_data.shape)

plt.figure(figsize=(24, 8))
plt.plot(xlabel='Convolutional Layer Output')
plt.title('Visualize CNN')
plt.imshow(input_img_data)
#*******************************************************************************************************
