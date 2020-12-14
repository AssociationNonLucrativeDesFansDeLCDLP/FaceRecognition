# FaceRecognition
Projet de reconnaissance faciale par apprentissage automatisé pour traçage de bouding boxes et identification des acteurs du groupe Les Nuls qu'on ne présente plus - Alain Chabat, Dominique Farrugia et Chantal Lauby - au sein du ~~film~~chef-d'œuvre d'Alain Berberian ; La Cité De La Peur. 

Nous nous appuyons sur la librairie OpenCV et sa technologie de CascadeClassifier pour la détection de visages et sur TensorFlow pour la création d'un modèle neuronal prédictif de classification des visages.

Nous avons créé un dataset composé de visages issus du film, le training set contient des visages d'une séquence au début du film et le validation set contient des visages d'une séquence à la fin du film.

Nous avons 4 classes représentant 4 personnes : Alain Chabat, Dominique Farrugia, Chantal Lauby et Gerard Darmon.

Le jeu de données est augmenté afin de rendre le modèle plus robuste, nous passons d'une centaine de capture de visages par acteur à plusieurs milliers grâce à des méthodes de rotation, de zoom, de changement de luminosité et de symétrie verticale.

Au total nous entrainons le model sur 16000 visages et nous le validons sur environ 1600.

Les visages sont redimensionnés en 128 par 128 pixels et converties en nuances de gris.

Le modèle de classification est un CNN, il est composé comme suit: 
  - Conv2D(32, 3, activation='relu', input_shape=self.input_shape)
  - Conv2D(32,3, strides=(3,3), activation='relu')
  - MaxPooling2D(pool_size=(2,2))
  - Dropout(.25)
  - Conv2D(32, 3, activation='relu', input_shape=self.input_shape)
  - Conv2D(32,3, strides=(3,3), activation='relu')
  - MaxPooling2D(pool_size=(2,2))
  - Dropout(.25)
  - Flatten()
  - Dense(self.n_dense, activation='relu')
  - Dropout(.5)
  - Dense(self.n_classes, activation='softmax')
 
 Nous avons designé le modèle en nous inspirant de solutions déjà existantes, nous avons ensuite procédé à une hyperparametrisation de deux paramètres, le nombre de neurones du Dense layer, et la profondeur des Convolution layers.
    
Le modèle final obtient une accuracy de 88% sur le validation set.
Les résultats sur des séquences du film sont cohérents dans l'ensemble.

Points à améliorer : 
  - Les visages avec des lunettes de soleil sont difficiles à classifier.
  - Le temps d'entrainement est très long, il faudrait utiliser une machine plus puissante ou installer Cuda pour réaliser l'entrainement sur une plus grande periode.
  - La sortie du modèle est souvent trop confiante, peu imoporte l'entrée nous avons une valeur élevée de confiance dans la prédiction. Ceci fait que lorsque l'algorithme de détection des visages se trompe nous avons des situations non voulues (une chaise classifiée comme étant Chantal Lauby par exemple).




Auguste Cousin et Erwan Le Pluard
