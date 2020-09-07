# train.py

# Import detecto libs, the lib is great and does all the work
# https://github.com/alankbi/detecto
from detecto import core
from detecto.core import Model

# Load all images and XML files from the Classification section
# or point it at "images" if you classified your own
dataset = core.Dataset('images_classified/')

# We initalize the Model and map it to the label we used in labelImg classification
model = Model(['aboriginal_flag'])

# The model.fit() method is the bulk of this program
# It starts training your model synchronously (the lib doesn't expose many logs)
# It will take up quite a lot of resources, and if it crashes on your computer
# you will probably have to rent a bigger box for a few hours to get this to run on.
# Epochs essentially means iterations, the more the merrier (accuracy) (up to a limit)
# It will take quite a while (15 mins?) for this process to end, grab a wine.
# If you get process killed, this process is memory bound. Try increase your disk swap ram.
model.fit(dataset, epochs=3, verbose=True)

# TIP: The more images you classify and the more epochs you run, the better your results will be.

# Once the model training has finished, we can save to a single file.
# Passs this file around to anywhere you want to now use your newly trained model.
model.save('model.pth')

# If you have got this far, you've already trained your very own unique machine learning model
# What are you going to do with this new found power?
