import numpy as np

def prepare_image(image_raw):
  try:
    return np.expand_dims(image_raw, axis=0)
  except Exception as e:
    raise e.add_note(
      "Error while preparing image"
    )