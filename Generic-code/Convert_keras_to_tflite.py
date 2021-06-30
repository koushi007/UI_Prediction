import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('Generic_0.991') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('Generic_model.tflite', 'wb') as f:
  f.write(tflite_model)