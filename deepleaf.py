import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import re

# Define the path to the dataset
dataset_path = '/kaggle/input/color'  # Replace with the actual path

# Set image size
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Function to load images, labels, and severities
def load_images_labels_severity(dataset_path):
    images = []
    labels = []
    severities = []
    class_names = set()  # Use set to avoid duplicates

    for root, dirs, files in os.walk(dataset_path):
        for name in dirs:
            class_names.add(re.sub(r'___\d+$', '', name))  # Extract class name without severity
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png"):
                label = os.path.basename(root)
                img_path = os.path.join(root, name)
                print(f"Processing file: {img_path}")  # Debugging statement
                try:
                    img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
                    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
                    images.append(np.array(img))
                    labels.append(re.sub(r'___\d+$', '', label))  # Remove severity from label
                    severity_match = re.search(r'___(\d+)$', label)
                    if severity_match:
                        severities.append(int(severity_match.group(1)))
                    else:
                        severities.append(0)  # Default severity if not found
                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    severities = np.array(severities)

    return images, labels, severities, list(class_names)

# Load data
images, labels, severities, class_names = load_images_labels_severity(dataset_path)

print(f"Number of images loaded: {len(images)}")  # Debugging statement
print(f"Class names: {class_names}")  # Debugging statement

# Display some images, their labels, and severities
def display_sample_images(images, labels, severities, class_names):
    if len(images) == 0:
        print("No images to display.")  # Debugging statement
        return

    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(images))):  # Ensure we don't go out of bounds
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"{labels[i]} (Severity: {severities[i]})")
        plt.axis('off')
    plt.show()

display_sample_images(images, labels, severities, class_names)

# Convert labels to numerical values
label_to_index = {label: index for index, label in enumerate(class_names)}
index_to_label = {index: label for label, index in label_to_index.items()}
numeric_labels = np.array([label_to_index[label] for label in labels])

# Save class names to a CSV file
class_names_df = pd.DataFrame(class_names, columns=['Class Name'])
class_names_df.to_csv('class_names.csv', index=False)

# Save images, labels, and severities to files
np.save('images.npy', images)
np.save('labels.npy', numeric_labels)
np.save('severities.npy', severities)
print('Preprocessing complete. Class names saved to class_names.csv.')
print('Images, labels, and severities saved to images.npy, labels.npy, and severities.npy.')import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import re

# Define the path to the dataset
dataset_path = '/kaggle/input/color'  # Replace with the actual path

# Set image size
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Function to load images, labels, and severities
def load_images_labels_severity(dataset_path):
    images = []
    labels = []
    severities = []
    class_names = set()  # Use set to avoid duplicates

    for root, dirs, files in os.walk(dataset_path):
        for name in dirs:
            class_names.add(re.sub(r'___\d+$', '', name))  # Extract class name without severity
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png"):
                label = os.path.basename(root)
                img_path = os.path.join(root, name)
                print(f"Processing file: {img_path}")  # Debugging statement
                try:
                    img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
                    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
                    images.append(np.array(img))
                    labels.append(re.sub(r'___\d+$', '', label))  # Remove severity from label
                    severity_match = re.search(r'___(\d+)$', label)
                    if severity_match:
                        severities.append(int(severity_match.group(1)))
                    else:
                        severities.append(0)  # Default severity if not found
                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    severities = np.array(severities)

    return images, labels, severities, list(class_names)

# Load data
images, labels, severities, class_names = load_images_labels_severity(dataset_path)

print(f"Number of images loaded: {len(images)}")  # Debugging statement
print(f"Class names: {class_names}")  # Debugging statement

# Display some images, their labels, and severities
def display_sample_images(images, labels, severities, class_names):
    if len(images) == 0:
        print("No images to display.")  # Debugging statement
        return

    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(images))):  # Ensure we don't go out of bounds
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"{labels[i]} (Severity: {severities[i]})")
        plt.axis('off')
    plt.show()

display_sample_images(images, labels, severities, class_names)

# Convert labels to numerical values
label_to_index = {label: index for index, label in enumerate(class_names)}
index_to_label = {index: label for label, index in label_to_index.items()}
numeric_labels = np.array([label_to_index[label] for label in labels])

# Save class names to a CSV file
class_names_df = pd.DataFrame(class_names, columns=['Class Name'])
class_names_df.to_csv('class_names.csv', index=False)

# Save images, labels, and severities to files
np.save('images.npy', images)
np.save('labels.npy', numeric_labels)
np.save('severities.npy', severities)
print('Preprocessing complete. Class names saved to class_names.csv.')
print('Images, labels, and severities saved to images.npy, labels.npy, and severities.npy.')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

X = images
y = numeric_labels
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding='same', name='conv_1', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='pool_1'))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides=(1,1), padding='same', name='conv_2', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='pool_2'))

model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides=(1,1), padding='same', name='conv_3', activation='relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units = 512, name='fc_1', activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 256, name='fc_2', activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

# The number of units here shall be equal to the number of classes
model.add(tf.keras.layers.Dense(units = 38, name='fc_3', activation='softmax'))

tf.random.set_seed(1)
model.build(input_shape=(None, 128, 128, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(train_generator, epochs=15, validation_data=(X_test, y_test) , shuffle=True)

import matplotlib.pyplot as plt
import numpy as np
hist = history.history
x_arr = np.arange(len(hist['loss'])) + 1

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,2,1)
ax.plot(x_arr, hist['loss'], '-o',label='Training Loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation Loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1,2,2)
ax.plot(x_arr, hist['accuracy'], '-o',label='Training Accuracy')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation Accuracy')
ax.legend(fontsize=15)
plt.show()

num_images_to_display =3
random_indices = np.random.choice(len(X_test), num_images_to_display, replace=False)

predictions = model.predict(X_test[random_indices])
predicted_classes = np.argmax(predictions, axis=1)
plt.figure(figsize=(15, 10))
for i, index in enumerate(random_indices):
    plt.subplot(3, 1, i + 1)
    plt.imshow(X_test[index])
    plt.title(f"Predicted: {class_names[predicted_classes[i]]}, Severity:{severities[index]} , Actual: {class_names[y_test[index]]}")
    plt.axis('off')
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

image_dir = '/content/drive/MyDrive/Grape___Black_rot'

# Extract features from all images
features_list = []
image_paths = []
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    features = extract_features(img_path, model)
    features_list.append(features)
    image_paths.append(img_path)

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)
# Convert features list to numpy array
features_array = np.array(features_list)
pca = PCA(n_components=0.95)  # Keep 95% of the variance
features_pca = pca.fit_transform(features_scaled)

num_clusters = 5  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features_pca)

# Save results to a DataFrame
results = pd.DataFrame({'ImagePath': image_paths, 'Cluster': clusters})
results.to_csv('clustered_images.csv', index=False)
print("Clustering completed. Results saved to 'clustered_images.csv'.")

def plot_tsne(features_pca, clusters):
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)

    results_tsne = pd.DataFrame({'Dimension1': features_tsne[:, 0],
                                 'Dimension2': features_tsne[:, 1],
                                 'Cluster': clusters})

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_tsne, x='Dimension1', y='Dimension2', hue='Cluster', palette='viridis')
    plt.title('t-SNE Visualization of Clusters')
    plt.show()

plot_tsne(features_pca, clusters)

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = '/content/clustered_images.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Iterate through each row in the CSV file
for index, row in data.iterrows():
    image_path = row[0]  # Path to the image
    tag = row[1]         # Corresponding tag

    # Load the image
    image = Image.open(image_path)

    # Plot the image and tag
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'Tag: {tag}')
    plt.axis('off')  # Hide axes
    plt.show()

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, ConfusionMatrixDisplay, PrecisionRecallDisplay
import seaborn as sns

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Precision, recall, F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

plt.figure(figsize=(15, 10))
for i, class_name in enumerate(class_names):
    PrecisionRecallDisplay.from_predictions(y_test == i, y_pred[:, i], name=class_name, ax=plt.gca())
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
