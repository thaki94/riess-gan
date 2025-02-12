import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_test_data(test_data_path, en_test_target_path=None, ss_test_target_path=None):
    img_list = os.listdir(test_data_path)

    if en_test_target_path is None or ss_test_target_path is None:
        # Only load test_data
        test_data = []

        for image_name in img_list:
            t_data_path = os.path.join(test_data_path, image_name)
            test_data.append(cv2.cvtColor(cv2.imread(t_data_path), cv2.COLOR_BGR2RGB))

        test_data = np.array(test_data)
        print(len(test_data), 'files loaded from -', test_data_path)

        return test_data

    else:
        # Load test_data, en_test_target, and ss_test_target
        test_data = []
        en_test_target = []
        ss_test_target = []

        for image_name in img_list:
            t_data_path = os.path.join(test_data_path, image_name)
            en_target_path = os.path.join(en_test_target_path, image_name)
            ss_target_path = os.path.join(ss_test_target_path, image_name)

            # Read images
            data = cv2.imread(t_data_path)
            en_target = cv2.imread(en_target_path)
            ss_target = cv2.imread(ss_target_path)

            # Convert images to RGB
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            en_target = cv2.cvtColor(en_target, cv2.COLOR_BGR2RGB)
            ss_target = cv2.cvtColor(ss_target, cv2.COLOR_BGR2RGB)

            # Resize images to (256, 256)
            data = cv2.resize(data, (256, 256))
            en_target = cv2.resize(en_target, (256, 256))
            ss_target = cv2.resize(ss_target, (256, 256))

            # Append to respective lists
            test_data.append(data)
            en_test_target.append(en_target)
            ss_test_target.append(ss_target)

        # Convert lists to numpy arrays
        test_data = np.array(test_data)
        en_test_target = np.array(en_test_target)
        ss_test_target = np.array(ss_test_target)

        print(len(test_data), 'files loaded from:', test_data_path)

        return test_data, en_test_target, ss_test_target

def normalize(X,range_max=1,range_min=-1,convert_to_image=False):

  min=np.min(X)
  max=np.max(X)


  X_std = (X - min) / (max- min)
  X_scaled = X_std * (range_max - range_min) + range_min

  if convert_to_image:

  	X_scaled=np.array(X_scaled,dtype=np.uint8)

  return X_scaled


def save_results(save_path,en_predicted_target,ss_predicted_target):

	try:
	    os.mkdir(save_path)
	except Exception as e:
	    print(e)

	en_predicted_target_save_path=os.path.join(save_path,'en_predicted_target')
	ss_predicted_target_save_path=os.path.join(save_path,'ss_predicted_target')

	try:
	    os.mkdir(en_predicted_target_save_path)
	except Exception as e:
	    print(e)

	try:
	    os.mkdir(ss_predicted_target_save_path)
	except Exception as e:
	    print(e)

	for i in range(len(en_predicted_target)):

	    cv2.imwrite(os.path.join(en_predicted_target_save_path,str(i)+'.png'),cv2.cvtColor(en_predicted_target[i],cv2.COLOR_BGR2RGB))
	    cv2.imwrite(os.path.join(ss_predicted_target_save_path,str(i)+'.png'),cv2.cvtColor(ss_predicted_target[i],cv2.COLOR_BGR2RGB))

	print(len(en_predicted_target), 'files saved to:', save_path)

def visualize(test_data,en_test_target=None,en_predicted_target=None,ss_test_target=None,ss_predicted_target=None):

	num_samples = len(test_data)

	if(en_test_target is None or ss_test_target is None):
		
		for i in range(num_samples):

			plt.figure(figsize=(7.7, 2.6))  # Adjusted figure size for 1 row and 5 columns

			# Plot test data (input image)
			plt.subplot(1, 3, 1)
			plt.imshow(test_data[i])
			plt.title(f'Test Image')
			plt.axis('off')

			# Plot en_test_target (ground truth for first output)
			plt.subplot(1, 3, 2)
			plt.imshow(en_predicted_target[i])
			plt.title('Predicted EN Target')
			plt.axis('off')

			# Plot en_predicted_target (predicted first output)
			plt.subplot(1, 3, 3)
			plt.imshow(ss_predicted_target[i])
			plt.title('Predicted SS Target')
			plt.axis('off')


			plt.suptitle(f'Sample {i+1}')
			plt.tight_layout()

			# Show the plot
			plt.show()
			plt.close()
	
	else:

		for i in range(num_samples):

			plt.figure(figsize=(20, 4))  # Adjusted figure size for 1 row and 5 columns

			# Plot test data (input image)
			plt.subplot(1, 5, 1)
			plt.imshow(test_data[i])
			plt.title(f'Test Image')
			plt.axis('off')

			# Plot en_test_target (ground truth for first output)
			plt.subplot(1, 5, 2)
			plt.imshow(en_test_target[i])
			plt.title('EN Test Target (Ground Truth)')
			plt.axis('off')

			# Plot en_predicted_target (predicted first output)
			plt.subplot(1, 5, 3)
			plt.imshow(en_predicted_target[i])
			plt.title('Predicted EN Target')
			plt.axis('off')

			# Plot ss_test_target (ground truth for second output)
			plt.subplot(1, 5, 4)
			plt.imshow(ss_test_target[i])
			plt.title('SS Test Target (Ground Truth)')
			plt.axis('off')

			# Plot ss_predicted_target (predicted second output)
			plt.subplot(1, 5, 5)
			plt.imshow(ss_predicted_target[i])
			plt.title('Predicted SS Target')
			plt.axis('off')

			plt.suptitle(f'Sample {i+1}')
			plt.tight_layout()

			# Show the plot
			plt.show()
			plt.close()
