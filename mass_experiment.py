import os
import cv2
from keras.models import load_model
from keras.models import model_from_json
import glob
import numpy as np
import matplotlib.pyplot as plt

## GENERATE IMAGES

'''
os.system('rm -r experiment_unlabeled')
os.system('mkdir experiment_unlabeled')

for num_1 in range(10): #10
	for num_2 in range(10):
		#no duplicates
		if num_1 < num_2:

			#print(num_1, num_2)

			os.system('mkdir experiment_unlabeled/Nums'+str(num_1)+str(num_2))

			# have to skip 0 and 100. 

			for split in range(10, 100, 10):

				split = float(split)/100

				os.system('mkdir experiment_unlabeled/Nums'+str(num_1)+str(num_2)+'/split'+str(split))
				os.system('rm -r checkpoint')
				os.system('python3 main.py --dataset mnist --input_height=28 --output_height=28 --train --image_split='+str(split)+ ' --mnist_1='+str(num_1)+ ' --mnist_2='+str(num_2))
'''

## RUN 5x (0, x) for x in 1,..,9
'''
os.system('rm -r experiment_avg')
os.system('mkdir experiment_avg')

for trial in range(5):

	os.system('mkdir experiment_avg/trial'+str(trial))

	for num_1 in range(1,10):

		os.system('mkdir experiment_avg/trial'+str(trial)+'/Nums0'+str(num_1))

		for split in range(10, 100, 10):

			split = float(split)/100

			os.system('mkdir experiment_avg/trial'+str(trial)+'/Nums0'+str(num_1)+'/split'+str(split))
			os.system('rm -r checkpoint')

			directory = 'experiment_avg/trial'+str(trial)
			os.system('python3 main.py --dataset mnist --input_height=28 --output_height=28 --train --image_split='+str(split)+ ' --mnist_1=0'+ ' --mnist_2='+str(num_1)+ ' --output_dir='+directory)
'''

## RUN 5x (0, x) for x in 1,..,9
'''
os.system('rm -r infogan_experiments2')
os.system('mkdir infogan_experiments2')

for trial in range(3):

	os.system('mkdir infogan_experiments2/trial'+str(trial))

	#for num_1 in range(1,10):
	num_1 = 5

	os.system('mkdir infogan_experiments2/trial'+str(trial)+'/Nums0'+str(num_1))

	for split in range(10, 100, 10):

		split = float(split)/100

		os.system('mkdir infogan_experiments2/trial'+str(trial)+'/Nums0'+str(num_1)+'/split'+str(split))
		os.system('rm -r checkpoint')

		directory = 'infogan_experiments2/trial'+str(trial)
		os.system('python3 main.py --dataset mnist --input_height=28 --output_height=28 --train --infogan=True --image_split='+str(split)+ ' --mnist_1=0'+ ' --mnist_2='+str(num_1)+ ' --output_dir='+directory)
'''

## RUN CELEB EXPERIMENT: 
'''
for split in range(40, 50, 10):

	print(split)

	if split == 50:
		continue

	split = float(split)/100

	os.system('rm -r checkpoint')
	os.system('mkdir celeb_experiment/split'+str(split))
	os.system('python3 main.py --dataset celebA --input_height=108 --train --crop --image_split='+str(split))
'''

## CLASSIFY IMAGES

#load mnist model.. 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

model_name = 'mnist_model'
json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_name+".h5")

model.summary()


'''
def plot_stuff(num_1, num_2):
	xs = [0.0]
	ys = [0.0]

	for split in range(10, 100, 10):

		split = float(split)/100

		#read images and format

		directory = '/home/valerie/DCGAN-tensorflow/infogan_experiments1/trial0/Nums05/split'+str(split)

		#directory = 'experiment_unlabeled/Nums'+str(num_1)+str(num_2)+'/split'+str(split)

		X_data = []
		files = glob.glob(directory+'/*.png')

		if len(files) == 0:
			xs.append(0)
			ys.append(0)
			print(num_1, num_2, "error")
			continue

		for myFile in files:
		    image = cv2.imread(myFile)
		    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		    image = np.reshape(gray, (28, 28, 1))

		    X_data.append(image)

		X_data = np.asarray(X_data)
		
		#classify images..
		pred = model.predict(X_data)


		count = 0
		total = 0
		count1 = 0
		total1 = 0

		for i in range(len(files)):
			#print(files[i][len(files[i])-7:])

			if files[i][len(files[i])-5] == '0':
				total = total + 1

			if files[i][len(files[i])-5] == '1':
				total1 = total1 + 1

			if files[i][len(files[i])-5] == '0'  and pred[i,num_2] == 1:
				count = count + 1
			elif files[i][len(files[i])-5] == '1'  and pred[i,num_1] == 1:
				count1 = count1 + 1

		print(count, total-count, total)
		print(total1-count1, count1, total1)

		#temp = pred[:,num_1] == 1
		

		num1s = len(pred[(pred[:,num_1] == 1)])
		num2s = len(pred[(pred[:,num_2] == 1)])

		xs.append(split)

		if num1s+num2s == 0:
			ys.append(0)
		else:
			ys.append(float(num1s)/float(num1s+num2s))
			print(split, float(num1s)/float(num1s+num2s), num1s, num2s)

		#look at the ones misclassified??

	xs.append(1.)
	ys.append(1.)

	#plt.scatter(xs,ys)
	#plt.plot(xs, ys)

	#plt.savefig('newest_plots/nums'+str(num_1)+str(num_2)+'.png')

	#plt.close()

plot_stuff(0,5)
'''

###PLOT INDIVIDUAL AND AVERAGES:

xs = [0.0]

for split in range(10, 100, 10):
	split = float(split)/100
	xs.append(split)
xs.append(1.0)
print(xs)
num_1 = 0

for num_2 in range(1,10):

	if num_2 != 5:
		continue

	print('num:', num_2)

	avg_ys = [0.0]*11

	avg_count = [0.0]*9 #corresponds to the 0s... 
	avg_count1 = [0.0]*9
	avg_total = [0.0]*9
	avg_total1 = [0.0]*9

	for trial in range(3):

		print('trial:', trial)

		xs = [0.0]
		ys = [0.0]

		ctr = 0

		for split in range(10, 100, 10):

			split = float(split)/100
			directory = 'infogan_experiments1/trial'+str(trial)+'/Nums'+str(num_1)+str(num_2)+'/split'+str(split)

			#directory = 'experiment_avg/trial'+str(trial)+'/Nums'+str(num_1)+str(num_2)+'/split'+str(split)

			X_data = []
			files = glob.glob(directory+'/*.png')

			if len(files) == 0:
				xs.append(0)
				ys.append(0)
				print(num_1, num_2, "error")
				continue

			for myFile in files:
			    image = cv2.imread(myFile)
			    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			    image = np.reshape(gray, (28, 28, 1))

			    X_data.append(image)

			X_data = np.asarray(X_data)
			
			#classify images..
			pred = model.predict(X_data)

			num1s = len(pred[(pred[:,num_1] == 1)])
			num2s = len(pred[(pred[:,num_2] == 1)])

			zeroaszero = 0
			zerosasone = 0

			fiveaszero = 0
			fiveasone = 0

			for i in range(len(files)):

				if pred[i,num_1] == 1: #assigned as 0
					if files[i][len(files[i])-5] == '0':
						zeroaszero += 1
					else:
						zerosasone += 1

				elif pred[i, num_2] == 1: #classified as 5
					if files[i][len(files[i])-5] == '0':
						fiveaszero += 1
					else:
						fiveasone += 1

			print('zero', zeroaszero, zerosasone)
			print('five', fiveaszero, fiveasone)

			# for i in range(len(files)):
			# 	if files[i][len(files[i])-5] == '0':
			# 		avg_total[ctr] = avg_total[ctr] + 1

			# 	if files[i][len(files[i])-5] == '1':
			# 		avg_total1[ctr] = avg_total1[ctr] + 1

			# 	if files[i][len(files[i])-5] == '0'  and pred[i,num_2] == 1:
			# 		#avg_count[ctr] = avg_count[ctr] + 1
			# 	elif files[i][len(files[i])-5] == '1'  and pred[i,num_1] == 1:
			#		#avg_count1[ctr] = avg_count1[ctr] + 1


			xs.append(split)

			if num1s+num2s == 0:
				ys.append(0)
			else:
				ys.append(float(num1s)/float(num1s+num2s))
				print(split, float(num1s)/float(num1s+num2s), num1s, num2s)

			ctr = ctr + 1

		xs.append(1.)
		ys.append(1.)

		for i in range(11):
			avg_ys[i] += 0.33*ys[i]

		#print(len(xs), len(avg_ys))

	plt.plot(xs, avg_ys, c='b', ls='-')
	plt.plot(xs, xs, c='r', ls='-')
	# plt.scatter(xs,avg_ys)
	# plt.plot(xs, avg_ys)
	# plt.scatter(xs,xs)
	# plt.plot(xs,xs)
	#plt.savefig('unequalc1.png')
	#plt.savefig('avg_plots_infogan/nums'+str(num_1)+str(num_2)+'.png')
	plt.close()

	print(avg_count)
	print(avg_total)
	#print(avg_count1)
	#print(avg_total1)


# xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ys = [0.412, 0.3584, 0.3652, 0.3852, 0.3506, 0.2934, 0.431, 0.3834, 0.4104]

# plt.plot(xs, ys, c='b', ls='-')
# plt.plot(xs, xs, c='r', ls='-')
# plt.savefig('celeb.png')
# plt.close()

#plot_stuff(0,2)
'''
#iterate through folders
for num_1 in range(10): #10
	for num_2 in range(10):
		#no duplicates
		if num_1 < num_2:

			print(num_1, num_2)
			#append (0,0) and (1., 1.) at end, plot and save graph.

			plot_stuff(num_1, num_2)

			#print(xs, ys)
'''

				




