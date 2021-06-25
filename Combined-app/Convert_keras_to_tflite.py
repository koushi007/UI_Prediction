import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('Combined-toy0.976') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('Combined_toy_model.tflite', 'wb') as f:
  f.write(tflite_model)