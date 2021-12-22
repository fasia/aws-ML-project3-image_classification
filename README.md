# Udacity project #3: Dog breed Image Classification with AWS Sagemaker

The dataset used in this project is available at https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
We use pre-trained model Resnet-50 with a depth of 50 layers as our model of choice. 
The steps for the image classification in this project:

1. First we download the dataset. 
2. unzip the folder
3. check the content
4. upload the folder to S3
5. create a hyperparameter tuning range for ResNet50 retrained model.
6. run tuner.fit() with entry point of hpo.py (configuration of the Resnet, training and testing the model with the givern hyperparameters)
7. get the hyperparameter of the best training job in tuner.
8. add profiling and debugging to the train model for logging info in cloudwatch
9. use the hyperparameters of the best training job and run estimator with model_train (train a new model with the given hpo)
10. plot the output 
11. use the estimator for deploying an endpoint that predicts the dog breed from the train model.
12. select a test image and call the endpoint with the test image.
13. check if the model predict the dog breed accurately. 
