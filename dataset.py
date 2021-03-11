 
import pandas as pd
from os import path
import os
import torch
from transformers import DistilBertTokenizerFast
import  numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor

import base_navigator


MAIN_DIR = './data/images'
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
navigator = base_navigator.BaseNavigator()
device = torch.device('cpu')
image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).to(device)

import attr

@attr.s
class NavigationDataset:
    """Construct a NavigationDataset dataset.
    `train` is the train split.
    `dev` is the dev split.
    `test` is the test split.
    """
    train = attr.ib()
    dev = attr.ib()
    test = attr.ib()

    @classmethod
    def from_NavigationSplit(cls, train, dev, test):
        """Construct a TextGeoDataset."""
        return NavigationDataset(
            train,
            dev,
            test
        )

    @classmethod
    def load(cls, dataset_path):
        logging.info("Loading dataset from <== {}.".format(dataset_path))

        dataset_navigation = torch.load(dataset_path)

        return dataset_navigation

    @classmethod
    def save(cls, dataset_navigation, dataset_path):

        torch.save(dataset_navigation, dataset_path)
        logging.info("Saved data to ==> {}.".format(dataset_path))



class NavigationSplit(torch.utils.data.Dataset):
    """A split of of the Touchdown dataset.

    `route_panoids`: The list of panorama ids of the route from start to end.
    `navigation_text`: The instruction text for navigation.

    """

    def __init__(self, data, split, calc_image_features=False):
        self.route_panoids = data['route_panoids']
        self.end_heading = data['end_heading']
        self.split = split

        self.navigation_text = tokenizer(
            data['navigation_text'].tolist(), truncation=True,
            padding=True, add_special_tokens=True)


        start_heading = 0
        self.all_list_action_numeric = []
        self.all_binary_action = []
        for idx, route_panoids in enumerate(self.route_panoids):
            actions, list_action_numeric, panoids_list, execution_list, binary_actions = \
                navigator.get_actions(route_panoids, start_heading)
            self.all_list_action_numeric.append(list_action_numeric)
            self.all_binary_action.append(binary_actions)

            if calc_image_features:
                self.calc_image_features(idx, panoids_list)

    def calc_image_features(self, idx, panoids_list):
        list_images_features = []
        path_save = path.abspath(path.join(MAIN_DIR, self.split, str(idx) + ".pt"))
        if path.exists(path_save):
            return
        for img_idx, image_id in enumerate(panoids_list):
            feature_path = path.abspath(path.join(MAIN_DIR, 'features', image_id + '.pt'))
            image_features = torch.load(feature_path)
            list_images_features.append(image_features)
        torch.save(list_images_features, path_save)



    def __getitem__(self, idx: int):
        '''Supports indexing such that NavigationSplit[i] can be used to get
        i-th sample.
        Arguments:
          idx: The index for which a sample from the dataset will be returned.
        Returns:
          A single sample including text, route and start_heading.
        '''
        text = {key: torch.tensor(val[idx])
                for key, val in self.navigation_text.items()}

        # Read image features.
        path_image = path.abspath(path.join(MAIN_DIR, self.split, str(idx) + ".pt"))
        images_features = torch.cat(torch.load(path_image), axis=0).squeeze(0)

        torch_actions = torch.tensor(self.all_list_action_numeric[idx])
        torch_actions_binary = torch.tensor(self.all_binary_action[idx]).float()
        return {'text': text,
                'images_features': images_features,
                'actions': torch_actions,
                'actions_binary': torch_actions_binary,
                'route_panoids': self.route_panoids[idx]
                }

    def __len__(self):
        return len(self.navigation_text['input_ids'])


class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['images_features'].shape[0], reverse=True)
        # Get each sequence and pad it
        images = [x['images_features'] for x in sorted_batch]
        images_padded = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)
        lengths = [len(x) for x in images]
        # print ("this is {}".format(lengths))
        lengths = torch.LongTensor(lengths)
        actions = [torch.tensor(x['actions']) for x in sorted_batch]
        actions_padded = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True)
        route_panoids =  [x['route_panoids'] for x in sorted_batch]

        text = {key: x['text'][key] for x in sorted_batch for key, val in x['text'].items()}

        return {'text': text,
                'images_features': images_padded,
                'actions': actions_padded,
                'lengths': lengths,
                'route_panoids': route_panoids}


def create_dataset(
                  data_dir: str,
):
    dev_path = path.join(data_dir, "dev.json")
    test_path = path.join(data_dir, "test.json")
    train_path = path.join(data_dir, "train.json")

    dev_ds = pd.read_json(dev_path, lines=True)
    test_ds = pd.read_json(test_path, lines=True)
    train_ds = pd.read_json(train_path, lines=True)

    dev_ds_proc = NavigationSplit(dev_ds, "dev")
    test_ds_proc = NavigationSplit(test_ds, "test")
    train_ds_proc = NavigationSplit(train_ds, "train")

    return NavigationDataset.from_NavigationSplit(train_ds_proc, dev_ds_proc, test_ds_proc)
