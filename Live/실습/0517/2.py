import numpy as np
import tensorflow as tf
from Visualize import Visualize

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)

def preprocess():

    # MNIST 데이터 세트를 불러옵니다.
    mnist = tf.keras.datasets.mnist

    # MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images, train_labels = train_images[:5000], train_labels[:5000]
    test_images, test_labels = test_images[:1000], test_labels[:1000]

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = tf.expand_dims(train_images, -1)
    test_images = tf.expand_dims(test_images, -1)

    train_labels = tf.one_hot(train_labels, depth=10)
    test_labels = tf.one_hot(test_labels, depth=10)

    return train_images, test_images, train_labels, test_labels


def MLP():
# 지시사항 1: 지시사항의 구조를 보고 MLP함수를 완성하세요
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    # 분류기 (classifier)
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

# 지시사항 2: 지시사항의 구조를 보고 CNN함수를 완성하세요
def CNN():

    model = tf.keras.Sequential()
    
    # Feature Extractor
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="SAME",
            input_shape=(28, 28, 1),
        )
    )
    model.add(tf.keras.layers.MaxPool2D(padding="SAME"))
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="SAME"
        )
    )
    model.add(tf.keras.layers.MaxPool2D(padding="SAME"))
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="SAME"
        )
    )
    model.add(tf.keras.layers.MaxPool2D(padding="SAME"))
    model.add(tf.keras.layers.Flatten())
    
    # 분류기 (classifier)
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))


    return model

def main():

    train_images, test_images, train_labels, test_labels = preprocess()
    cnn_model = CNN()
    cnn_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    mlp_model = MLP()
    mlp_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]

    )
    
    # MLP모델부터 학습을 시작합니다.
    history_mlp = mlp_model.fit(
        train_images,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_images, test_labels),
    )
    
    # Test 테이터로 mlp 모델을 평가합니다.
    mlp_loss, mlp_test_acc = mlp_model.evaluate(test_images, test_labels, verbose=0)
    
    # CNN 모델 학습을 시작합니다.
    history_cnn = cnn_model.fit(
        train_images,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_images, test_labels),
    )
    
    # Test 테이터로 cnn 모델을 평가합니다.
    cnn_loss, cnn_test_acc = cnn_model.evaluate(test_images, test_labels, verbose=0)


    print("\nMLP Test Loss : {:.4f} | Test Accuracy : {:.4f}%".format(mlp_loss, mlp_test_acc*100))

    print("CNN Test Loss : {:.4f} | Test Accuracy : {:.4f}%".format(cnn_loss, cnn_test_acc*100))
    
    # 그래프로 CNN모델과 MLP 모델의 성능을 비교합니다.
    Visualize([("CNN", history_cnn), ("MLP", history_mlp)], "accuracy")


    return history_cnn
        
if __name__ == "__main__":
    main()