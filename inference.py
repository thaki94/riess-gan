#Load GAN models from /models

from modules.gan import RIESS_GAN
from modules.utils import *

image_shape = (256,256,3)

en_model_path= 'models/en_gan_good_weather.h5'
ss_model_path= 'models/ss_gan_good_weather.h5'

g_model = RIESS_GAN(image_shape,en_model_path,ss_model_path)

#Testing the dataset 1 - Good Weather Seen Sequence

test_data_path='sample_testing_data/dataset_1/test_data'

en_test_target_path='sample_testing_data/dataset_1/en_test_target' #ground truth for visualization and compare with the predictions
ss_test_target_path='sample_testing_data/dataset_1/ss_test_target' #ground truth for visualization and compare with the predictions

test_data,en_test_target,ss_test_target=load_test_data(test_data_path,en_test_target_path,ss_test_target_path)

test_data_norm=normalize(test_data)

en_predicted_target,ss_predicted_target=g_model.predict(test_data_norm)

en_predicted_target_norm=normalize(en_predicted_target,range_max=255,range_min=0,convert_to_image=True)
ss_predicted_target_norm=normalize(ss_predicted_target,range_max=255,range_min=0,convert_to_image=True)

visualize(test_data,en_test_target,en_predicted_target_norm,ss_test_target,ss_predicted_target_norm)

save_path='results/dataset_1'

save_results(save_path,en_predicted_target_norm,ss_predicted_target_norm)

#Testing the dataset 2 - Good Weather Unseen Sequence

image_shape = (256,256,3)

en_model_path= 'models/en_gan_all_weather.h5'
ss_model_path= 'models/ss_gan_all_weather.h5'

g_model = RIESS_GAN(image_shape,en_model_path,ss_model_path)

test_data_path='sample_testing_data/dataset_2/test_data'

en_test_target_path='sample_testing_data/dataset_2/en_test_target' #ground truth for visualization and compare with the predictions
ss_test_target_path='sample_testing_data/dataset_2/ss_test_target' #ground truth for visualization and compare with the predictions

test_data,en_test_target,ss_test_target=load_test_data(test_data_path,en_test_target_path,ss_test_target_path)

test_data_norm=normalize(test_data)

en_predicted_target,ss_predicted_target=g_model.predict(test_data_norm)

en_predicted_target_norm=normalize(en_predicted_target,range_max=255,range_min=0,convert_to_image=True)
ss_predicted_target_norm=normalize(ss_predicted_target,range_max=255,range_min=0,convert_to_image=True)

visualize(test_data,en_test_target,en_predicted_target_norm,ss_test_target,ss_predicted_target_norm)

save_path='results/dataset_2'

save_results(save_path,en_predicted_target_norm,ss_predicted_target_norm)

#Testing the dataset 3 - Snow (No Ground Truth Data)

test_data_path='sample_testing_data/dataset_3/test_data'

test_data=load_test_data(test_data_path)

test_data_norm=normalize(test_data)

en_predicted_target,ss_predicted_target=g_model.predict(test_data_norm)

en_predicted_target_norm=normalize(en_predicted_target,range_max=255,range_min=0,convert_to_image=True)
ss_predicted_target_norm=normalize(ss_predicted_target,range_max=255,range_min=0,convert_to_image=True)

visualize(test_data,en_predicted_target=en_predicted_target_norm,ss_predicted_target=ss_predicted_target_norm)

save_path='results/dataset_3'

save_results(save_path,en_predicted_target_norm,ss_predicted_target_norm)

#Testing the dataset 4 - Rain (No Ground Truth Data)

test_data_path='sample_testing_data/dataset_4/test_data'

test_data=load_test_data(test_data_path)

test_data_norm=normalize(test_data)

en_predicted_target,ss_predicted_target=g_model.predict(test_data_norm)

en_predicted_target_norm=normalize(en_predicted_target,range_max=255,range_min=0,convert_to_image=True)
ss_predicted_target_norm=normalize(ss_predicted_target,range_max=255,range_min=0,convert_to_image=True)

visualize(test_data,en_predicted_target=en_predicted_target_norm,ss_predicted_target=ss_predicted_target_norm)

save_path='results/dataset_4'

save_results(save_path,en_predicted_target_norm,ss_predicted_target_norm)