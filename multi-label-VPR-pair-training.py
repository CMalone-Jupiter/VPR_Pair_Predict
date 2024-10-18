import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from sklearnex import patch_sklearn
patch_sklearn()
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import itertools
import argparse
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('CUDA availability: ' + str(torch.cuda.is_available()))


parser = argparse.ArgumentParser()
parser.add_argument('-tr','--TrainDir', required=True, help='Train directory', type=str)
parser.add_argument('-va','--ValidDir', required=True, help='Validation directory', type=str)
parser.add_argument('-te','--TestDir', required=True, help='Test directory', type=str)
parser.add_argument('-s','--single', required=False, default=0, help='If using single techs', type=int)

# Define the dataset structure for pytorch
class MultiLabel_VPR(Dataset):
    def __init__(self, data_path, anno_path, num_refs=128, single=0, n_comps=None, not_training=1):
        # Define the techniques which are being considered in multi-process fusion
        #  These techniques have been chosen and used because they occur in the VPR-Bench framework
        techs = ['CoHOG', 'CALC', 'NetVLAD', 'RegionVLAD', 'AMOSNet', 'HybridNet', 'HOG', 'AlexNet_VPR', 'Ap-GeM', 'denseVLAD']
        if not single:
            # Define all possible pair combinations from the list of VPR techniques (order not important)
            possible_pairs = list(itertools.combinations(range(len(techs)), 2))
            # Define in string form each of the techniques
            self.classes = [techs[np.array(i)[0]]+'_'+techs[np.array(i)[1]] for i in possible_pairs]
        else:
            self.classes = techs

        # Limit the classes which are being considered to only those which include NetVLAD
        self.classes = [self.classes[1], self.classes[9], self.classes[17], self.classes[18],
                        self.classes[19], self.classes[20], self.classes[21], self.classes[22], self.classes[23]]
        
        # print('Num Classes: {}'.format(len(self.classes)))
        # print('Classes: {}'.format(self.classes))

        # Append a None class for when none of the fused VPR pairs can localize an image
        self.classes.append('None')

        # Find the filenames for the queries and annotations for which VPR pairs will successfully localize
        sample_names = [filename for filename in sorted(os.listdir(data_path))]
        # Annotations will be a multi-hot vector of which VPR techniques will successfully localize
        annotation_names = [filename for filename in sorted(os.listdir(anno_path))]

        self.imgs = []
        self.annos = []
        self.annos_numpy = []
        self.data_path = data_path
        self.data_filenames = []

        print('loading', anno_path)

        for sample in tqdm(range(0,len(sample_names))):

            # Reducing the multi-hot annotation vector to only relevant VPR pairs
            # Currently assumed to be csv file but can be changed to other file types
            anno = np.loadtxt(anno_path+annotation_names[sample], delimiter=',').astype(int)
            idxs = np.array([1, 9, 17, 18, 19, 20, 21, 22, 23])
            anno = anno[idxs]

            # For training data we only want to use samples where at least 1
            #  VPR pair will successfully localize an image
            if (np.sum(anno) > 0) or not_training:

                self.data_filenames.append(sample_names[sample])

                # Load precomputed VPR descriptors from the base technique.
                #  This vector is a concatentation of the query descriptor
                #  and at least the top VPR reference match according to the
                #  baseline technique
                img_path = os.path.join(self.data_path, sample_names[sample])
                img = np.load(img_path)

                # Compute the difference between the query descriptor and the 
                #  top VPR match reference descriptor
                #  n_comps is the number of feature components being used (default 128)
                img1_diff = img[1:num_refs+1,0:n_comps] - img[0,0:n_comps]

                # Flattening to make sure dimensions are correct
                full_desc = img1_diff.flatten().astype(float)
                self.imgs.append(torch.from_numpy(full_desc).float())

                # Append a 1 to the end of the annotations vector if none of the VPR pairs
                #  will succesfully localize the query
                if np.sum(anno) == 0:
                    anno = np.append(anno, 1)
                else:
                    anno = np.append(anno, 0)

                self.annos_numpy.append(anno.astype(float))
                self.annos.append(torch.from_numpy(anno).float())



    def __getitem__(self, item):
        anno = self.annos[item]
        img = self.imgs[item]
        filename = self.data_filenames[item]
        # score = self.scores[item]
        return img, anno, filename#, score

    def __len__(self):
        return len(self.imgs)

# Define the simple MLP model to be used
class Multilabel_Classifier(nn.Module):
    def __init__(self, n_classes, layer_size, num_layers, dropout, pca_size, num_refs=1):
        super(Multilabel_Classifier, self).__init__()
        self.num_layers = num_layers

        self.input_layer =  nn.Sequential(nn.Linear(in_features=(pca_size)*(num_refs+1), out_features=layer_size), nn.Dropout(p=dropout))
        self.hidden = nn.ModuleList()
        if self.num_layers > 1:
            for k in range(self.num_layers-1):
                self.hidden.append(nn.Sequential(nn.Linear(in_features=layer_size, out_features=layer_size), nn.Dropout(p=dropout)))
        self.output = nn.Linear(in_features=layer_size, out_features=n_classes)


    def forward(self, x):

        y = F.relu(self.input_layer(x))     
        for layer in self.hidden:
            y = F.relu(layer(y))

        y = self.output(y)
        return y

# Define validation loop
def validate_model(this_dataloader, this_model, this_criterion):
    with torch.no_grad():

        the_losses = []
        for imgs, batch_targets, filename in this_dataloader:

            imgs, batch_targets = imgs.to(device), batch_targets.to(device)
            model_batch_result = this_model(imgs)
            val_loss = this_criterion(model_batch_result, batch_targets.type(torch.float))

            the_losses.append(val_loss.item())

    valid_loss_value = np.mean(the_losses)

    return valid_loss_value

# Here is an auxiliary function for checkpoint saving.
def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

# Here is the main training loop
# It's mostly a standard fully connected network training process
def train_model(config, train_dir, valid_dir, test_dir, train_annotations, valid_annotations, test_annotations, single):

    num_workers = 0

    device = torch.device('cuda')
    # Save path for checkpoints
    save_path = 'Path to checkpoint save directory'

    # Create pytorch datasets out of the precomputed features and labels
    train_dataset = MultiLabel_VPR(train_dir+'Path to features/', train_annotations, config['num_refs'], single, config['pca_size'], not_training=0)
    valid_dataset = MultiLabel_VPR(valid_dir+'Path to features/', valid_annotations, config['num_refs'], single, config['pca_size'], not_training=0)
    tester_dataset = MultiLabel_VPR(test_dir+'Path to features/', test_annotations, config['num_refs'], single, config['pca_size'], not_training=1)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(tester_dataset, batch_size=config['batch_size'], num_workers=num_workers)


    num_train_batches = int(np.ceil(len(train_dataset) / config['batch_size']))

    # Initialize the model
    model = Multilabel_Classifier(len(train_dataset.classes), config['layer_size'], config['num_layers'], config['dropout'], config['pca_size'], num_refs=config['num_refs'])
    # Switch model to the training mode and move it to GPU.
    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, min_lr = config['lr']/10)

    # If more than one GPU is available we can use both to speed up the training.
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    os.makedirs(save_path, exist_ok=True)
    # Loss function
    #  The BCE logit loss function includes a sigmoid activation
    #  so this is not needed in the model
    criterion = nn.BCEWithLogitsLoss() 


#################################################################### Training ########################################################################
    print('Starting Training')

    # Run training
    epoch = 0
    iteration = 0
    valid_loss_value = 1

    while True:
        model_result_all = []
        targets_all = []
        batch_losses = []
        train_filenames = []

        for imgs, targets, filename in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            model_result = model(imgs)
            loss = criterion(model_result, targets.type(torch.float))
            loss.backward()
            optimizer.step()
            

            batch_losses.append(loss.item())
            model_result_all.extend(model_result.detach().cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
            train_filenames.extend(np.array(list(filename)))

            iteration += 1

        loss_value = np.mean(batch_losses)
        
        print("epoch:{:2d} iter:{:3d} train loss:{:.3f}".format(epoch, iteration, loss_value))
        
        if epoch % config['test_freq'] == 0:
            model.eval()
            valid_loss_value = validate_model(valid_dataloader, model, criterion)
            test_loss_value = validate_model(test_dataloader, model, criterion)
            model.train()

        scheduler.step(valid_loss_value)
            
        for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

        if curr_lr < config['lr']:
            break

        epoch += 1
        if config['max_epoch_number'] < epoch:
            break


def main():

    opt = parser.parse_args()
    train_dir = opt.TrainDir
    valid_dir = opt.ValidDir
    test_dir = opt.TestDir

    # Initialize the dataloaders for training.
    valid_annotations = valid_dir+'Path to multi-hot labels/'
    train_annotations = train_dir+'Path to multi-hot labels/'
    test_annotations = test_dir+'Path to multi-hot labels/'

    # Some default values to use which were found for the paper results
    parameters_dict = {'max_epoch_number': 100, 'test_freq': 1, 'save_freq': 30, 'batch_size': 8,
    'lr': 0.00045503245783766623, 'layer_size': 32, 'num_layers': 3, 'dropout': 0.126450391535889,
    'pca_size': 128, 'num_refs': 1, 'num_labels': 50}

    train_model(parameters_dict, train_dir, valid_dir, test_dir, train_annotations, valid_annotations, test_annotations, opt.single)

if __name__ == "__main__":
    main()