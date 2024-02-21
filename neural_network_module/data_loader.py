'''Python Script to use Pytorch's built-in DataLoader and Dataset classes to load and preprocess our data.'''

from __future__ import print_function, division, with_statement
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#### Dataset class to create an object to store all the samples, process then, batch and shuffle them using using the inbuilt Pytorch Dataset Class. 
class metric_nn_dataset(Dataset):
    """Dataset used to train and test the Grasp Metric Neural Network"""

### Constructor to initialize an object of the class metricNNDataset:
    def __init__(self, x_data, y_data, transform=None):
        """"
        Input Arguments:
            x_csv_file(string): Path to the csv file which contains the datapoints.
            y_csv_file(string): Path to the csv file which contains the labels corresponding to the datapoints.
            root_dir(string): Directory which contains all the required files.
        """

        self.x_data = x_data
        self.y_data = y_data
        self.idx = []

        self.transform = transform

### Overriding the __len__ method so that it returns the size of the dataset.
    def __len__(self):
        #return len(self.datapoints)
        return len(self.x_data)

### Overriding the __getitem__ method so that it can support indexing of our dataset.
# This is not able to read the first datapoint
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapoint = self.x_data[idx]
        label = self.y_data[idx]
        sample = {'X':datapoint, 'Y':label}

        # This conditional is important otherwise we cannot transform the datapoint and labels to tensors.
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


### A callable class to convert the datapoint and label from the sample into tensors.
# This implements a transformation that can be applied on the data. 
class to_tensor(object):
    '''Convert the ndarrays in a sample to Tensors'''
# We just need to implement the __call__ method here
    def __call__(self, sample):
        datapoint, label = sample['X'], sample['Y']
        new_shape = (1)
        label = torch.tensor(label).float()
        return {'X': torch.from_numpy(datapoint).float(), 
                'Y': label.view(new_shape)}


