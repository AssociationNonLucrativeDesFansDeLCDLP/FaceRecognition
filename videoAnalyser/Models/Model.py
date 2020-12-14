from tensorflow.keras import Sequential, layers, utils, models
from sklearn.model_selection import KFold
import numpy as np
import csv

class Model(object):
	"""docstring for Model"""
	model=None
	history=None
	name=None
	writeHeader=True
	
	validation_split=0.2
	n_classes=8
	input_shape=(100, 100, 1)
	
	epochs = 5
	n_dense=128
	depth=2

	eval=None
		
	def __init__(self, name):
		super(Model, self).__init__()
		self.name=name
		print(f'instantiating model : {str(name)}')
		
	def buildModel(self, input_shape, n_classes, n_dense, depth):
		self.input_shape=input_shape
		self.n_classes=n_classes
		self.n_dense=n_dense
		self.depth=depth

		model=Sequential()
		for i in range(depth):
			model.add(layers.Conv2D(32*(i+1), 3, activation='relu', input_shape=self.input_shape))
			model.add(layers.Conv2D(32*(i+1),3, strides=(3,3), activation='relu'))
			model.add(layers.MaxPooling2D(pool_size=(2,2)))
			model.add(layers.Dropout(.25))
		#model.add(layers.Conv2D(64, 3, activation='relu'))
		#model.add(layers.Conv2D(64, 3, strides=(3,3), activation='relu'))
		#model.add(layers.MaxPooling2D(pool_size=(2,2)))
		#model.add(layers.Dropout(.25))
		model.add(layers.Flatten())
		model.add(layers.Dense(self.n_dense, activation='relu'))
		model.add(layers.Dropout(.5))
		model.add(layers.Dense(self.n_classes, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		self.model=model

	def train(self, dataset, epochs=None):
		if (epochs):
			self.epochs=epochs
		if (self.model):
			history=self.model.fit(dataset.getXTrain(), dataset.getYTrain(), epochs=self.epochs, validation_data=(dataset.getXTest(), dataset.getYTest()), shuffle=True)
			self.history=history
		else:
			print('No model was built')
	
	def trainAllData(self, dataset, epochs=epochs):
		if (self.model):
			self.history=self.model.fit(dataset.getX(), dataset.getY(), epochs=epochs)
		else:
			print('Build model first')
	

	def trainKFold(self, dataset, epochs, K, n_dense, depth, file):
		kfold = KFold(n_splits=K, shuffle=True)
		fold_no = 1
		for train, test in kfold.split(dataset.getX(), dataset.getY()):
			print(f'Fold number : {fold_no}')
			self.buildModel(dataset.getInputShape(), n_classes=dataset.getNClasses(), n_dense=n_dense, depth=depth)
			self.history = self.model.fit(dataset.getX()[train], dataset.getY()[train], epochs=epochs, validation_data=(dataset.getX()[test], dataset.getY()[test]))
			self.eval = self.model.evaluate(dataset.getX()[test], dataset.getY()[test], verbose=1)
			self.saveKfold(file, writeHeader=fold_no==1, fold=fold_no)
			fold_no+=1
			

	def evaluate(self, dataset):
		if (self.model):
			self.eval=self.model.evaluate(dataset.getXTest(), dataset.getYTest())
		else:
			print('build model first')
	
	def saveResult(self, file, writeHeader=False):
		print('saving history for model : '+str(self.name))
		with open(file, 'a', newline='') as csvfile:
			fieldnames = ['name', 'n_epochs', 'n_dense', 'depth', 'eval', 'accuracy', 'val_accuracy', 'loss', 'val_loss']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			if (writeHeader):
				writer.writeheader()
			writer.writerow({'name':str(self.name),
							 'n_epochs': str(self.epochs), 
							 'n_dense': str(self.n_dense),
							 'depth': str(self.depth),
							 'eval':str(self.eval),
							 'accuracy': self.history.history['accuracy'], 
							 'val_accuracy': self.history.history['val_accuracy'], 
							 'loss': self.history.history['loss'], 
							 'val_loss': self.history.history['val_loss']})
	
	def saveKfold(self, file, writeHeader=False, fold=1):
		print('saving Kfold '+str(self.name))
		with open(file, 'a', newline='') as csvfile:
			fieldnames=['name', 'fold', 'n_epochs', 'n_dense', 'depth', 'eval', 'accuracy', 'val_accuracy', 'loss', 'val_loss']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			if(writeHeader):
				writer.writeheader()
			writer.writerow({'name':str(self.name),
							 'fold':str(fold),
							 'n_epochs': str(self.epochs), 
							 'n_dense': str(self.n_dense),
							 'depth': str(self.depth),
							 'eval':str(self.eval),
							 'accuracy': self.history.history['accuracy'], 
							 'val_accuracy':self.history.history['val_accuracy'],
							 'loss': self.history.history['loss'],
							 'val_loss': self.history.history['val_loss']})

	def accByClass(self, dataset, file, writeHeader=False):
		print('computing accuracy by class')
		data=[[] for x in range(dataset.getNClasses())]
		label=[[] for x in range(dataset.getNClasses())]
		for i in range(len(dataset.getY())):
			data[np.argmax(dataset.getY()[i])].append(dataset.getX()[i])
			label[np.argmax(dataset.getY()[i])].append(dataset.getY()[i])
		preds=[]
		print('here ?')
		for i in data:
			print(len(i))
		if(self.model):
			for i in range(dataset.getNClasses()):
				d=[np.expand_dims(t, axis=0) for t in data[i]]
				print(d.shape)
				preds.append(self.model.evaluate(d), label[i])
		print('there')
		for j in preds:
			print(f'preds for elements : {np.mean(j)}')
	
	def saveModel(self, file):
		if(self.model):
			self.model.save(file)
		else:
			print('Build Model first')
	
	def loadModel(self, file):
		self.model=models.load_model(file)
		
			
	