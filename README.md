# FakeSafe - Encode Private Information into Fake Message Unrecognizable to Humans via Multi-Step Adversarial Network

FakeSafe is a project which trains privacy preserving generative model that enables information transfer across data silos without compromising privacy and protect at both technological and human levels. This is the code implementation which demonstrates our idea of FakeSafe project, which can be viewed on [GitHub](https://github.com/mikemikezhu/fake-safe).

## Instructions

### Setup the Project

First and foremost, we need to setup our project by downloading the required dataset and installing the necessary third-party libraries. Developers may run the following bash script to setup the project.

```shell
sh setup.sh
```

**Parameters:**

The first parameter of the command specifies whether it is necessary to download the dataset.

For the time being, developers might probably need to download `eng.txt`, which is required to encode and decode the English text, and `face.zip`, the dataset which is used to encode and decode the human face images.

If developers specify a `1` in the first parameter of the command, the dataset `eng.txt` will be downloaded into `data` folder. However, considering that `face.zip`, as part of the medical dataset, requires developers to conduct special training before getting the full access to it, please kindly contact [liudianbo@gmail.com](liudianbo@gmail.com) or [12051594@life.hkbu.edu.hk](12051594@life.hkbu.edu.hk) for more information about the dataset.

```shell
sh setup.sh 1
```

Nevertheless, it might be unnecessary to download the dataset everytime when developers run the project. Then the extra parameter `1` shall be ignored.

### Run Classifier

Before developers run the project, the classifier models are required to train and generate so as to perform evaluation and training metrics for the project.

The following script is to run the classifier models.

```shell
sh run_classifier.sh
```

**Parameters:**

(1) The first parameter specifies the script name.

If developers wish to run all of the classifiers by default, they may specify `0` as the first parameter.

On the contrary, if developers wish to run the specific classifier, they may specify the file name of the classifier as the first parameter. For example, if developers wish to run the MNIST classifier, they need to specify `classifier_mnist`, which is the file name, as the first parameter.

```shell
# Run all of the classifiers
sh run_classifier.sh 0 (second_param) (third_param)

# Run MNIST classifier
sh run_classifier.sh classifier_mnist (second_param) (third_param)
```

Here are the list of classifiers which developers may train and generate.

```shell
# Classifier for face in RGB
classifier_face_rgb
# Classifier for face in greyscale
classifier_face
# Classifier for fashion
classifier_fashion
# Classifier for MNIST
classifier_mnist
```

(2) The second parameter specifies whether or not to directly display the output of training.

If developers wish to display the output directly, they may specify `1` as the second parameter. Otherwise, they may specify `0` as the second parameter.

```shell
# Display the output directly
sh run_classifier.sh (first_param) 1 (third_param)
# Do not display the output directly
sh run_classifier.sh (first_param) 0 (third_param)
```

(3) The third parameter controls whether or not to save the output to file.

If developers wish to save the output to file, they may specify `1` as the second parameter. Otherwise, they may specify `0` as the second parameter.

```shell
# Save the output to file
sh run_classifier.sh (first_param) (second_param) 1
# Do not save the output to file
sh run_classifier.sh (first_param) (second_param) 0
```

### Run FakeSafe Encoding and Decoding

Developers may run the FakeSafe encoding and decoding by running the following commands.

```shell
sh run_fake_safe.sh
```

**Parameters:**

(1) The first parameter specifies the script name. For example, if developers wish to run the Face -> MNIST -> Face, they need to specify `one_layer_face_mnist`, which is the file name, as the first parameter.

```shell
sh run_fake_safe.sh one_layer_face_mnist (second_param) (third_param)
```

Here are the list of cases which developers may use to encode and decode.

```shell
# One layer
# Face -> Face -> Face (RGB)
one_layer_face_face_rgb
# Face -> MNIST -> Face (RGB)
one_layer_face_mnist_rgb
# Face -> MNIST -> Face
one_layer_face_mnist
# Fashion -> Fashion -> Fashion
one_layer_fashion_fashion
# Fashion -> MNIST -> Fashion
one_layer_fashion_mnist
# MNIST -> Fashion -> MNIST
one_layer_mnist_fashion
# MNIST -> MNIST -> MNIST
one_layer_mnist_mnist
# Word -> Fashion -> Word
one_layer_word_fashion
# Word -> Fashion -> Word (Use embedding)
one_layer_word_fashion_embedding
# Word -> Word -> Word (Use embedding)
one_layer_word_word_embedding
# Sentence -> Fashion -> Sentence
one_layer_sentence_fashion
# Sentence -> Fashion -> Sentence (Use word-to-word embedding)
one_layer_sentence_fashion_embedding

# Two layer
# Face -> Fashion -> MNIST -> Fashion -> Face (RGB)
two_layer_face_fashion_mnist_rgb
# Face -> Fashion -> MNIST -> Fashion -> Face
two_layer_face_fashion_mnist
# Fashion -> MNIST -> MNIST -> MNIST -> Fashion
two_layer_fashion_mnist_mnist
# MNIST -> MNIST -> Fashion -> MNIST -> MNIST
two_layer_mnist_mnist_fashion

# Three layer
# Face -> MNIST -> Fashion -> Fashion -> Fashion -> MNIST -> Face (RGB)
three_layer_face_mnist_fashion_fashion_rgb
# Face -> MNIST -> Fashion -> Fashion -> Fashion -> MNIST -> Face
three_layer_face_mnist_fashion_fashion
# MNIST -> MNIST -> Fashion -> Fashion -> Fashion -> MNIST -> MNIST
three_layer_mnist_mnist_fashion_fashion
```

(2) The second parameter specifies whether or not to directly display the output of training.

If developers wish to display the output directly, they may specify `1` as the second parameter. Otherwise, they may specify `0` as the second parameter.

```shell
# Display the output directly
sh run_fake_safe.sh (first_param) 1 (third_param)
# Do not display the output directly
sh run_fake_safe.sh (first_param) 0 (third_param)
```

(3) The third parameter controls whether or not to save the output to file.

If developers wish to save the output to file, they may specify `1` as the second parameter. Otherwise, they may specify `0` as the second parameter.

```shell
# Save the output to file
sh run_fake_safe.sh (first_param) (second_param) 1
# Do not save the output to file
sh run_fake_safe.sh (first_param) (second_param) 0
```