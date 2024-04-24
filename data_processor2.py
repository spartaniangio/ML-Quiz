import pandas as pd  # Import pandas library for data manipulation
import ipaddress  # Import ipaddress library (not used in this code)
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder from scikit-learn for label encoding
import tensorflow as tf  # Import TensorFlow library
from sklearn.model_selection import train_test_split  # Import train_test_split function from scikit-learn

df = pd.read_csv("Darknet.csv")

df = df.drop(["Flow ID", "Timestamp", "Label2", "Src IP", "Dst IP"], axis=1)

df = df.dropna()

label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

df.to_csv("processed.csv", index=False)

df = pd.read_csv("processed.csv")

features = df.drop(['Label1'], axis=1)
label = df['Label1']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),  # Input layer with 250 neurons and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(4, activation='softmax')  # Output layer with 4 neurons and softmax activation (assuming 4 classes)
])

model.compile(optimizer='adam',  # Use Adam optimizer
              loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy loss for multi-class classification
              metrics=['accuracy'])  # Track accuracy metric

model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test))  # Train for 20 epochs with batch size 512, using validation data

with open('accuracy2.txt', 'w') as f:
    f.write(f'Test Loss: {loss:.4f}\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')
