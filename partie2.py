'''
MLP : Multi-Layer Perceptron 
(Multilayer Perceptron with 2 Hidden Layers )
https://github.com/nlintz/TensorFlow-Tutorials
Configuration
-Lors de l'apprentissage, utiliser l'erreur comme critere de comparaison entre la sortie desiree et la sortie reelle du reseau
-Fonction non lineaire des neurones de type sigmoide
-Utilisation du moment ("momentum")
-Configuration disposant de 2 couches cachees
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#timer pour temps de calcul
import timeit
import tensorflow as tf

# Nos parametres
# Valeur du taux d'apprentissage (optimisation et minimise la fonction de perte d'un reseau neuronal.)
learning_rate = 0.001 # determine a quelle vitesse ou a quelle vitesse vous souhaitez mettre a jour les parametres
# nombre d'iterations a l'entrainement
training_epochs = 30
# taille des lots
batch_size = 50
# Affichage des etapes
display_step = 1

# Parametre du reseau
# Nombre de neurones sur la premiere couche cachee
n_hidden_1 = 256
# Nombre de neurones sur la seconde couche cachee
n_hidden_2 = 256 
# MNIST donnee en entree (dimension de 28*28)
n_input = 784 
# MNIST total classes (0-9 chiffres)
n_classes = 10 

# tensorflow entree
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Non lineaire (tf.sigmoid, tf.tanh, tf.nn.elu, tf.nn.softplus, tf.nn.softsign)
# Lineaire (tf.nn.relu, tf.nn.relu6, tf.nn.crelu, tf.nn.relu_x)
# Creation du model avec double couche cachee et Fonction d'activation
def multilayer_perceptron(x, weights, biases):
    print("Activation couche fonction sigmoid")
    # Premiere couche cachee avec activation sigmoide 
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']) # matmul multiplication des matrices
    layer_1 = tf.sigmoid(layer_1)
    #layer_1 = tf.nn.relu(layer_1)

    # Deuxieme couche cachee avec activation sigmoide
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']) # matmul multiplication des matrices
    layer_2 = tf.sigmoid(layer_2)
    #layer_2 = tf.nn.relu(layer_2)

    # Couche de sortie avec activation sigmoide
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Poids
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
# Biais
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construction du model
pred = multilayer_perceptron(x, weights, biases)

#-------- COUT / PERTE cost--------
# Documentation : 
# https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/contrib/nn

# SOFTMAX (bon pour classification) <= A utiliser
# Mesure l'erreur de probabilite dans les taches de classification discretes dans lesquelles les classes s'excluent mutuellement
# Par exemple, chaque image CIFAR-10 est etiquetee avec une seule etiquette: une image peut etre un chien ou un camion, mais pas les deux.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# SIGMOID (bon pour regression, demande plus d'iterations)
# Mesure l'erreur de probabilite dans les taches de classification discretes dans lesquelles chaque classe est independante et ne s'excluent pas mutuellement. 
# Par exemple, on pourrait effectuer une classification multilingue ou une image peut contenir a la fois un Elephant et un chien en meme temps.
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

#-------- OPTIMISATION train --------
# Documentation : 
# http://sebastianruder.com/optimizing-gradient-descent/
# https://smist08.wordpress.com/2016/10/04/the-road-to-tensorflow-part-10-more-on-optimization/
def choix_apprentissage(apprentissage):
  '''
  Choix de notre apprentissage (train)
  '''
  if(apprentissage=="MOMENTUM") :
    # MOMENTUM
    # Utilisation de l'apprentissage avec l'algorithme Momentum (https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer#class_tftrainmomentumoptimizer)
    # momentum = Cette variable amortit la vitesse et reduit l'energie cinetique du systeme, sinon la particule ne s'arretera jamais au bas d'une colline (coef. de friction)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9).minimize(cost)
    return optimizer

  if(apprentissage=="ADAM") :
    # ADAM (Adaptive Moment Estimation)
    # Utilisation de l'apprentissage avec l'algorithme d'Adam (https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return optimizer

  if(apprentissage=="GRADIENT") :
    # GRADIENT
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    return optimizer

  if(apprentissage=="ADAGRAD") :
    # ADAGRAD
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
    return optimizer


#Choix de notre apprentissage ADAM MOMENTUM GRADIENT
choix = "ADAM"
title = "learning_rate:"+str(learning_rate)+", Taille blocs:"+str(batch_size)+ ", Nombre de neurones:"+str(n_hidden_1)+ " et "+str(n_hidden_2)
print("%s %s"%(choix,title))
print("Apprentissage:",choix)
optimizer = choix_apprentissage(choix)

#-------- Initialisation des variables
init = tf.global_variables_initializer()

#Pour la courbe matplorlib
accuracies = []
# Lancement
with tf.Session() as sess:
    sess.run(init)
    start_time = timeit.default_timer()
    # Cycle de d'apprentissage
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Boucle sur tous les lots
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Execute l'optimisation op (backprop) et le cout op (pour obtenir la valeur de perte)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Calcule la perte moyenne
            avg_cost += c / total_batch
        # Afficher les informations pour chaque etape d'une periode
        if epoch % display_step == 0:
            print("Periode:", '%04d' % (epoch+1), "Cout=", "{:.9f}".format(avg_cost))
            accuracies.append(avg_cost)
    print("----- Optimisation terminee -----")
    elapsed = timeit.default_timer() - start_time
    print("----- Temps ",elapsed, "secondes -----")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calcule de la precision
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("----- mnist.test -----")
    precision = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print("Precision: "+str(precision))
    print("----- mnist.validation -----")
    precision = accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels})
    print("Precision: "+str(precision))

#Courbe Matplotlib (Partie facultative)
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(accuracies)
fig.suptitle(choix+" "+title, fontsize=8)
plt.xlabel("Accuracy :" +str(precision), fontsize=8)
plt.ylabel('Epoch', fontsize=8)
fig.savefig('test.png')

plt.show()