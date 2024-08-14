# EatSmart-Calorie Optimization

This project empowers users to make informed dietary choices by offering a convenient and potentially more accurate way to estimate the calorie content of their meals. It utilizes image recognition and generative AI to simplify calorie tracking.

Image Recognition: Leverages pre-trained models like TensorFlow Lite's "mobilenet_v2" to identify food categories in pictures. Alternative: DenseNet201 model

Calorie Estimation: Provides an estimated calorie range based on the identified food category and a reference calorie database.

Generative AI Integration: Utilizes  to generate informative outputs beyond just a calorie range. Outputs could include: "Looks like a slice of apple (around 50 calories)."

Food-101 from Kaggle used for training the image recognition model.
