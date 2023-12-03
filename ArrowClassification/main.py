import tensorflow as tf
import pandas as pd
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class_names = {
            "North": 0,
            "North East": 1,
            "East": 2,
            "South East": 3,
            "South": 4,
            "South West": 5,
            "West": 6,
            "North West": 7}

reverse_class_names = {v: k for k, v in class_names.items()}

predictions = {}

def train_new_model():
    datasetPercentageForTraining = 0.7
    datasetPercentageForTesting = 0.3

    data = pd.read_csv('../ArrowGenerator/arrow_data.csv')
    dataCount = len(data)
    trainingCount = int(dataCount * datasetPercentageForTraining)
    testingCount = int(dataCount * datasetPercentageForTesting)

    trainingData = data[:trainingCount]
    testingData = data[trainingCount:trainingCount + testingCount]

    trainingImages = []
    testingImages = []

    trainingLabels = []
    testingLabels = []

    for index, row in trainingData.iterrows():
        image = cv2.imread(f"../ArrowGenerator/trainingImages/arrows/{row['Arrow Number']}")
        ret, thresholdhed_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
        trainingImages.append(image)
        trainingLabels.append(class_names[row['Direction']])
    print("Training data loaded")

    for index, row in testingData.iterrows():
        image = cv2.imread(f"../ArrowGenerator/trainingImages/arrows/{row['Arrow Number']}")
        ret, thresholdhed_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
        testingImages.append(image)
        testingLabels.append(class_names[row['Direction']])

    trainingImages = np.array(trainingImages)
    testingImages = np.array(testingImages)
    trainingLabels = np.array(trainingLabels)
    testingLabels = np.array(testingLabels)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(56, 56)),
        tf.keras.layers.Dense(8)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    epocs = int(input("Enter number of epocs: "))
    model.fit(trainingImages, trainingLabels, epochs=epocs)

    saveModel = input("Do you want to save the model? (y/n): ").strip().lower()
    if saveModel == 'y':
        model_name = input("Enter a name to save your model: ").strip()
        save_model(model, model_name)
    return model, testingImages, testingLabels


def save_model(model, model_name):
    model_path = f"models/{model_name}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_saved_model(model_name):
    model_path = f"models/{model_name}"
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model




def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(reverse_class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         reverse_class_names[true_label]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(8))
    plt.yticks([])
    thisplot = plt.bar(range(8), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')  # Adjust index for true_label if necessary

def get_test_data():
    # This part is repeated from your train_new_model function
    data = pd.read_csv('../ArrowGenerator/testing_arrow_data.csv')

    testingImages = []
    testingLabels = []

    for index, row in data.iterrows():
        image = cv2.imread(f"../ArrowGenerator/testingImages/arrows/{row['Arrow Number']}")
        ret, thresholdhed_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
        testingImages.append(image)
        testingLabels.append(class_names[row['Direction']])

    testingImages = np.array(testingImages)
    testingLabels = np.array(testingLabels)
    
    return testingImages, testingLabels

def predict_image(model, testingImages, testingLabels):
    test_loss, test_acc = model.evaluate(testingImages, testingLabels, verbose=2)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(testingImages)

    # for i in range(10):  # print first 10 predictions
    #     print("Predicted:", np.argmax(predictions[i]), "True Label:", testingLabels[i])

    failedImages = 0
    for i in range(len(predictions)):  # print first 10 predictions
        if np.argmax(predictions[i]) != testingLabels[i]:
            failedImages += 1
    print(f"Failed predictions: {failedImages}/{len(testingImages)}")


    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols



    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], testingLabels, testingImages)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], testingLabels)
    plt.tight_layout()
    plt.show()

def list_models():
    return [f for f in os.listdir('models/') if os.path.isdir(os.path.join('models/', f))]

def check_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
def main():
    choice = input("Do you want to load a saved model? (y/n): ").strip().lower()
    if choice == 'y':
        models = list_models()
        if not models:
            print("No saved models found.")
            return
        for idx, model_name in enumerate(models, 1):
            print(f"{idx}. {model_name}")
        selected_idx = int(input(f"Select a model (1-{len(models)}): "))
        selected_model_name = models[selected_idx - 1]
        model = load_saved_model(selected_model_name)
        testingImages, testingLabels = get_test_data()
    else:
        model, testingImages, testingLabels = train_new_model()

    predict_image(model, testingImages, testingLabels)


if __name__ == "__main__":
    check_gpu()
    main()
