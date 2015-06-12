import matplotlib.pyplot as plt
import cPickle

titre=raw_input('Nom du fichier a analyser  :\n')
nb=input('Numero de la couche a analyser  :\n')

f = file(titre, 'rb')
params = cPickle.load(f)
f.close()

if nb==3:
	print 'analysing layer3...'
	plt.hist(params[0].get_value(), 50, normed=1, facecolor='g', alpha=0.75)
	plt.title('layer 3')
if nb==2:
	print 'analysing layer2...'
	plt.hist(params[2].get_value(), 50, normed=1, facecolor='g', alpha=0.75)
	plt.title('layer 2')
if nb==1:	
	print 'analysing layer1...'
	a=[]
	for i in params[4].get_value():
		for j in i:
			for k in j:
				for l in k:
					a.insert(2,l)
	plt.hist(a, 100, normed=1, facecolor='g', alpha=0.75)
	plt.title('layer 1')

if nb==0:
	compteur=0;	
	print 'analysing layer0...'
	a=[]
	for i in params[6].get_value():
		for j in i:
			for k in j:
				for l in k:
					compteur=compteur+1
					a.insert(2,l)
	plt.hist(a, 100, normed=1, facecolor='g', alpha=0.75)
	plt.title('layer 0')

plt.show()


print compteur