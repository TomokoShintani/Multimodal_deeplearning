import os
import numpy as np
import pandas as pd
import random
import cv2
import torch
from torchvision import datasets, transforms, models
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt

table_data = pd.read_csv('../data/pokemon.csv')
image_directory = '../data/images/images/'

#-----データセットの作成-----#
image_df=[]

for filename in os.listdir(image_directory):
    if filename.endswith(".png"):
        image_name = filename.split('.')[0]

        image = cv2.imread(os.path.join(image_directory, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #np.array(image) : (H,W,RGB)

        #画像の名前と画像のデータがセットになったdictionaryを作成
        image_df.append({'Name':image_name, 'image':np.array(image)})
        
image_dataframe = pd.DataFrame(image_df)
        
#ポケモンの名前とTypeが入ってるtable_dataに画像データをくっつける
data = table_data.merge(image_dataframe, how = 'inner', on = 'Name')

# Grass, Fire, Waterの三値分類タスクにする
data = data.loc[data['Type1'].isin(['Grass', 'Fire', 'Water'])]

#技のデータを追加
moves_data = pd.read_csv('../data/metadata_pokemon_moves.csv')

water_moves_lst = moves_data.loc[moves_data['type']=='Water']['name'].tolist()
grass_moves_lst = moves_data.loc[moves_data['type']=='Grass']['name'].tolist()
fire_moves_lst = moves_data.loc[moves_data['type']=='Fire']['name'].tolist()

def assign_move(df):
    if df['Type1'] == 'Water':
        move = random.choice(water_moves_lst)
    elif df['Type1'] == 'Grass':
        move = random.choice(grass_moves_lst)
    elif df['Type1'] == 'Fire':
        move = random.choice(fire_moves_lst)
    return move

data['Move'] = data.apply(assign_move, axis=1)

mapping = {'Grass':0, 'Fire':1, 'Water': 2}
data['Type1'] = data['Type1'].map(mapping)

tokenizer = get_tokenizer("basic_english")
counter = Counter()


for move in data['Move']:
    counter.update(tokenizer(move))

vocabulary = vocab(counter, min_freq=4, specials=('<unk>', '<PAD>'))

vocabulary.set_default_index(vocabulary['<unk>'])

def text_transform(df):
    _text = df['Move']
    text = [vocabulary[token] for token in tokenizer(_text)]
    #text = [vocabulary['<BOS>']] + text + [vocabulary['<EOS>']]
    return text

def text_transform(df):
    _text = df['Move']
    text = [vocabulary[token] for token in tokenizer(_text)]
    #text = [vocabulary['<BOS>']] + text + [vocabulary['<EOS>']]
    return text

data['Move_ids'] = data.apply(text_transform, axis=1)

for lst in data['Move_ids']:
    if len(lst) < 2:
        lst.append(1)
    while len(lst) > 2:
        lst.pop(len(lst) - 1)

#-----image audmentation and create dataset-----#
# 計算高速化のため、arrayのarrayにする
image_array = np.array([np.array(img) for img in data['image']])

# Pytorchのモジュールの仕様上(..., H, W)の順番にする
image = np.transpose(torch.tensor(np.array(image_array.astype(np.int32))), (0, 3, 1, 2))

dataset = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                         torch.Tensor(data['Move_ids']),
                                         image/255.0)

transform_shift_1 = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.4, 0.4))])

dataset_shift_1 = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                                 torch.Tensor(data['Move_ids']),
                                                 transform_shift_1(image/255.0))

transform_shift_2 = transforms.Compose([transforms.RandomAffine(degrees=90, translate=(0.3, 0.3))])

dataset_shift_2 = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                                 torch.Tensor(data['Move_ids']),
                                                 transform_shift_2(image/255.0))

transform_zoom = transforms.Compose([transforms.CenterCrop(size=100),transforms.Resize(120)])

dataset_zoom = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                              torch.Tensor(data['Move_ids']),
                                              transform_zoom(image/255.0))

transform_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.5)])

dataset_flip = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                              torch.Tensor(data['Move_ids']),
                                              transform_flip(image/255.0))

transform_flip_shift = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.3), transforms.RandomAffine(degrees=0, translate=(0.3, 0.3))])

dataset_flip_shift = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                                    torch.Tensor(data['Move_ids']),
                                                    transform_flip_shift(image/255.0))

transform_rotate= transforms.Compose([transforms.RandomRotation(degrees = (-180, 180))])

dataset_rotate= torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                               torch.Tensor(data['Move_ids']),
                                               transform_rotate(image/255.0))

transform_rotate_shift = transforms.Compose([transforms.RandomRotation(degrees = (-180, 180)), transforms.RandomAffine(degrees=0, translate=(0.3, 0.3))])

dataset_rotate_shift = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                                      torch.Tensor(data['Move_ids']),
                                                      transform_rotate_shift(image/255.0))

transform_erase = transforms.Compose([transforms.RandomErasing(p = 0.8, scale = (0.02, 0.33), ratio = (0.3, 3.3))])

dataset_erase = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']),
                                               torch.Tensor(data['Move_ids']),
                                               transform_erase(image/255.0))

test_dataset = dataset

dataset_trainval = torch.utils.data.ConcatDataset([dataset_erase, dataset_zoom, dataset_rotate, dataset_rotate_shift, dataset_flip, dataset_shift_1, dataset_shift_2])

#train validationに分割
train_dataset, valid_dataset = torch.utils.data.random_split(dataset_trainval, [len(dataset_trainval)-100, 100])

batch_size = 4

#dataloader作る
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)