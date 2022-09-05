IMG_SHAPE=(224,224,3)
InceptionResnet=tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,weights='imagenet',input_shape=IMG_SHAPE)

def create_model_inceptionresnet(model_name):
  models = { "InceptionResnet" : InceptionResnet }
  model = models[model_name]
  for layer in model.layers:
    layer.trainable = True
  x = tf.keras.layers.Flatten()(model.output)
  x = tf.keras.layers.Dense(512, activation='relu')(x)
  x = tf.keras.layers.Dense(256,activation='relu')(x)
  x = tf.keras.layers.Dense(128,activation='relu')(x)
  x = tf.keras.layers.Dense(32,activation='softmax')(x)

  model = Model(inputs= model.input, outputs=x)

  
  my_model = tf.keras.models.clone_model(model)
  return my_model 
# model = tf.keras.Model(feature_extractor_model.input, x)
model=create_model_inceptionresnet('InceptionResnet')
model_flavia_inceptionresnet=model
model_flavia_inceptionresnet.compile( optimizer= tf.keras.optimizers.Adam(learning_rate=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath='/content/gdrive/MyDrive/Model_Paths/Flavia_PlantLeaf/InceptionResnet', verbose=2,save_best_only=True,monitor='val_accuracy',mode='max')
callbacks = [checkpoint]
history_flavia_inceptionresnet=model_flavia_inceptionresnet.fit(x = train,epochs=100,validation_data=validation,verbose=2, callbacks=callbacks)
def plot_loss_curves(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
reqd_img_flavia_inceptionresnet_rgb=plot_loss_curves(history_flavia_inceptionresnet_rgb)
