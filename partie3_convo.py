"""
Deep MNIST for Experts
Reseau a architecture profonde utilisation de CNN
CNN : Convulution Neural Network
Convulution Neural Network avec 2 couche)
https://www.tensorflow.org/get_started/mnist/pros
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf


sess = tf.InteractiveSession()

#Nous commencons a construire le graphique de calcul en creant des noeuds pour les images d'entree et les classes de sortie cible.
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#variable initialisation d'un tenseur avec des 0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

#Classe de prediction et fonction de perte
print("-- Similaire partie 1 ---")
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#nous pouvons evaluer notre precision sur les donnees de test. (92% correct)
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

########################################################
##### Creation d'un reseau convolutif multicouches 
##### Pour avoir 99% de precision 
########################################################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution et Pooling
def conv2d(x, W):
  #Calcule une convolution 2-D avec des tensions d'entree et de filtrage 4-D.
  h_conv0 = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') 
  return h_conv0

#Creation d'un regroupement maximal, Pooling layer downsamples by 2X.
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#---- I Premiere couche convolutionnelle ----
#Convolution suivie d'un max pooling
#La convolution calculera 32 fonctionnalites pour chaque piece de 5x5. Son tenseur de poids aura une forme de [5, 5, 1, 32]
W_conv1 = weight_variable([5, 5, 1, 32]) #Les deux premieres dimensions sont la taille du patch, voici le nombre de canaux d'entree et le dernier est le nombre de canaux de sortie
b_conv1 = bias_variable([32]) #Biais un composant pour chaque canal de sortie.

#Pour appliquer la couche, nous remodelons d'abord x a un tenseur 4d
#La deuxieme et troisieme dimensions correspondant a la largeur et a la hauteur de l'image, et la dimension finale correspondant au nombre de canaux de couleur.
x_image = tf.reshape(x, [-1,28,28,1]) #-1 = la taille de cette dimension est calculee de sorte que la taille totale reste constante

#Nous transformons ensuite x_image avec le tensor du poids, ajoutons le biais, appliquons la fonction ReLU et max_pool_2x2
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#La methode max_pool_2x2 reduira la taille de l'image a 14x14.
h_pool1 = max_pool_2x2(h_conv1) #The pooling ops sweep a rectangular window over the input tensor, computing a reduction operation for each window
#---------------------------------------

#---- II Deuxieme couche convolutionnelle ----
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#---------------------------------------

#---- III Couche a connexion dense ----
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Pour reduire la surutilisation (overfitting), nous appliquons dropout avant la couche de lecture
keep_prob = tf.placeholder("float") #Placeholder Pour la probabilite que la sortie d'un neurone soit conservee pendant le dropout.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #Gere automatiquement les sorties de neurones de mise a l'echelle en plus de les masquer, de sorte que le dropout fonctionne juste sans aucune mise a l'echelle supplementaire.

#Couche de lecture
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Apprentissage et Evaluation du model (prend 3 min sur un i7 a 3.2ghz)
#On utilise le parametre supplementaire keep_prob dans feed_dict pour controler le taux dropout
print("-- Amelioration de la precision ---")
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#Utilisation de l'optimisation ADAM
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("train accuracy %g"%train_accuracy)
#seg fault
#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#print("Precision:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


print("-- mnist.test ---")
for i in range(1000):
  batch = mnist.test.next_batch(200)
  if i%100 == 0:
    test_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, test accuracy %g"%(i, test_accuracy))

print("-- mnist.validation ---")
for i in range(1000):
  batch = mnist.validation.next_batch(200)
  if i%100 == 0:
    test_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, validation accuracy %g"%(i, test_accuracy))


