import tensorflow as tf, numpy as np, cv2, os, random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

image_size_x, image_size_y = 224, 224
encode = {'notumor': 0, 'meningioma': 1, 'glioma': 2, 'pituitary': 3}
decode = {0: 'notumor', 1: 'meningioma', 2: 'glioma', 3: 'pituitary'}

def preprocess_image(img_path):
    # read in the image
    if type(img_path) == np.ndarray:
        img = img_path
    else:
        img = cv2.imread(img_path)
        
    # convert image to grayscale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize the image array
    # img = img / 255

    # resize the image for uniformity
    img = cv2.resize(img, (image_size_x, image_size_y))

    return img

def load_data(dataset_path):
    x_data = [] # images
    y_data = [] # labels

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        # check if the path is a directory and skip if not
        if not os.path.isdir(folder_path):
            continue  
        
        for file in os.listdir(folder_path):
            # find path of the image
            file_path = os.path.join(folder_path, file)

            # preprocess the image
            img = preprocess_image(file_path)

            # append the new image to the x array
            x_data.append(img)

            # append the label to the y array
            y_data.append(encode[folder])
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # shuffle data for better training
    x_data, y_data = shuffle(x_data, y_data, random_state=101)
    
    return x_data, y_data

def predict_image(model, img):
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return decode[np.argmax(prediction)]

train_dataset = 'backend/model/dataset/training/' 
test_dataset = 'backend/model/dataset/testing/'

x_train, y_train = load_data(train_dataset)
x_test, y_test = load_data(test_dataset)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

model_path = 'backend/model/model.keras'
model = tf.keras.models.load_model(model_path)

fig, axes = plt.subplots(4, 5, figsize=(15, 10))
axes = axes.ravel()
for i in range(0, 20):
    index = random.randint(0, len(x_test))
    axes[i].imshow(x_test[index], cmap='gray')
    axes[i].set_title(f'Actual: {decode[np.argmax(y_test[index])]} \n Predicted: {predict_image(model, x_test[index])}')
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.5)
plt.show()