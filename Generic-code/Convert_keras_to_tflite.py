import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('Combined-toy0.998') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('Combined_toy_model998.tflite', 'wb') as f:
  f.write(tflite_model)