import warnings
from pathlib import Path
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

from utils import clear, count_sliding_window, make_optimizer, make_scheduler, CrossEntropy2d, accuracy, metrics, save_test, sliding_window, grouper, convert_from_color, convert_to_color, global_accuracy
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
# from focal_loss import FocalLoss
# from recal_loss import RecallLoss
from jaccard_ce_loss import JaccardCELoss

from models.LANet import LANet
def calc_alpha(curr_iter, all_iters, weight=1.0):
    r = (1.0-float(curr_iter)/all_iters)** 2.0
    return weight*r


class Trainer():
    
    def __init__(self, net, loader, params, scheduler = True, cbkp = None,):
        
        self.net = net
        self.loader = loader
        self.params = params
        self.iter_ = 0
        self.epoch_loss = []
        self.epoch_acc = []
        self.accuracies = []
        self.losses = np.zeros(1000000)
        self.mean_losses = np.zeros(100000000)
        self.print_each = params['print_each'] or 100 # Print statistics every 500 iterations
        
        # # Weights for class balancing
        self.weight_cls = self.prepare([self.params['weights']])
        # self.criterion = FocalLoss(mode='multiclass', alpha=0.5, gamma=2.0, reduction='mean')
        self.criterion = nn.CrossEntropyLoss(weight=self.weight_cls[0], )
        # self.criterion = JaccardCELoss()
        
        # Define an id to a trained model. Use the number of seconds since 1970
        time_ = str(time.time())
        time_ = time_.replace(".", "")
        self.model_id = time_

        # Create optimizer
        self.optimizer = make_optimizer(self.params['optimizer_params'], self.net)
        # Create scheduler
        self.scheduler = make_scheduler(self.params['lrs_params'], self.optimizer) if scheduler else None

        self.last_epoch = 0

        # Load a previously model if it exists
        if cbkp is not None:
            self.load(cbkp)

        Path(os.path.join(self.params['results_folder'])).mkdir(parents=True, exist_ok=True)
            
    def load(self, path):
        
        # Check if model file exists
        assert os.path.exists(path), "{} cant be opened".format(path)
        
        checkpoint = torch.load(path)

        try:
            self.last_epoch = checkpoint['epoch']
            self.model_id = checkpoint['model_id']
            self.losses = checkpoint['losses']
            self.mean_losses = checkpoint['mean_losses']
            self.epoch_loss = checkpoint['epoch_loss']
            self.epoch_acc = checkpoint['epoch_acc']
            self.iter_ = checkpoint['iter_']
            # self.acc_ = checkpoint['acc_']
        except KeyError as e:
            print(e)
            pass
            
        # Load model and optimizer params
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None:
            for _ in range(0, self.last_epoch): self.scheduler.step()

    def save(self, path = None):

        if path is None:
            # path = './{}_model_final.pth.tar'.format(self.model_id)
            path = os.path.join(self.params['results_folder'], f"{self.model_id}_model_final.pth.tar")

        # Save current loss, epoch, model weights and optimizer params
        torch.save({
            'epoch': self.last_epoch,
            'losses': self.losses,
            'mean_losses': self.mean_losses,
            'epoch_loss': self.epoch_loss,
            'epoch_acc': self.epoch_acc,
            'iter_': self.iter_,
            # 'acc_': self.acc_,
            'model_id': self.model_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def prepare(self, l, non_blocking=True):
        device = torch.device('cpu' if self.params['cpu'] is not None else 'cuda') # Define run-device torch
        def _prepare(tensor: torch.Tensor):
            if self.params['precision'] == 'half': tensor = tensor.half() # Convert to half precision
            # return tensor.to(device, non_blocking=non_blocking)
            return tensor.to(device, non_blocking=non_blocking)
        return [_prepare(_l) for _l in l]

    def test(self, test_loader = None, stride = None, window_size = None, batch_size = None, all=False):
    
        # Loading test parameters
        test_ld = test_loader if test_loader is not None else self.loader['test']
        assert test_ld is not None, "Test_loader can't be None"

        test_stride = stride if stride is not None else self.params['stride']
        assert test_stride is not None, "Stride not set"

        test_ws = window_size if window_size is not None else self.params['window_size']
        assert test_ws is not None, "Window size not set"

        bs = batch_size if batch_size is not None else self.params['bs']
        bs = bs if bs is not None else 1

        # Get all data in test data loader
        input_ids, label_ids = test_ld.dataset.get_dataset()
        # test_inputs_name = np.copy(input_ids)

        # Use the network on the test set
        test_images = (1 / 255 * np.asarray(io.imread(id)[:,:,:3], dtype='float32') for id in input_ids)
        test_labels = (np.asarray(io.imread(id)[:,:,:3], dtype='uint8') for id in label_ids)
        eroded_labels = (convert_from_color(io.imread(id)[:,:,:3]) for id in label_ids)
        
        all_preds = []
        all_gts = []
        
        # Switch the network to inference mode
        self.net.eval()
        
        for img, gt, gt_e, image_path in tqdm(zip(test_images, test_labels, eroded_labels, input_ids), total=len(input_ids), leave=False):
            # print(f'Image Path -> {image_path}')
            filepath = os.path.split(image_path)[1].split('.')[0]
            # print(filepath)
            # Path(f'./tmp/{filepath}').mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.params['results_folder'], 'inference', filepath)).mkdir(parents=True, exist_ok=True)
            
            pred = np.zeros(img.shape[:2] + (self.params['n_classes'],))

            total = count_sliding_window(img, step=test_stride, window_size=test_ws) // bs
            for i, coords in enumerate(tqdm(grouper(bs, sliding_window(img, step=test_stride, window_size=test_ws)), total=total, leave=False)):
                # Display in progress results
                if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = torch.argmax(pred, axis = -1)#np.argmax(pred, axis=-1) #Usar argmax do pytorch
                    fig = plt.figure()
                    
                    fig.add_subplot(1,3,1)
                    plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    fig.add_subplot(1,3,2)
                    plt.imshow(convert_to_color(_pred))
                    fig.add_subplot(1,3,3)
                    plt.imshow(gt)
                    # plt.show()
                    fig.savefig(f"./tmp/test_progress", dpi=fig.dpi, bbox_inches='tight')
                    plt.close(fig)
                        
                # Build the tensor (ex: 2048px/256px = 8 patches de 256x256)
                with torch.no_grad():
                    image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = Variable(torch.from_numpy(image_patches).cuda())
                
                # Do the inference
                if (type(self.net)==LANet):
                    outs,_ = self.net(image_patches)
                else:  
                     outs = self.net(image_patches)
                outs = outs.data.cpu().numpy()
                # trocar pred pela função do pytorch
                
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0)) # Usar transpose do Pytorch
                    pred[x:x+w, y:y+h] += out
                del(outs)

            pred = np.argmax(pred, axis=-1)

            # Display the result
            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax1.set_title('RGB')
            plt.imshow(np.asarray(255 * img, dtype='uint8'))
            ax2 = fig.add_subplot(1,3,2)
            ax2.set_title('Pred')
            plt.imshow(convert_to_color(pred))
            ax3 = fig.add_subplot(1,3,3)
            ax3.set_title('GT')
            plt.imshow(gt)
            # plt.show()
            fig.savefig(os.path.join(self.params['results_folder'], 'inference', filepath, 'result'), dpi=fig.dpi, bbox_inches='tight')
            plt.close(fig)

            all_preds.append(pred)
            all_gts.append(gt_e)

            # clear()
            # Compute some metrics
            
            # metrics(pred.ravel(), gt_e.ravel(), label_values=self.params['classes'], filepath=filepath)
        
        accuracy = metrics(
            predictions=np.concatenate([p.ravel() for p in all_preds]), 
            gts=np.concatenate([p.ravel() for p in all_gts]).ravel(),
            label_values=self.params['classes'],
            all=all,
            filepath=filepath
        )

        if all:
            save_test(acc=accuracy, all_preds=all_preds, all_gts=all_gts, 
                      path=os.path.join(self.params['results_folder'], 'segnet256_test_result.npz'))
            return accuracy, all_preds, all_gts
        else:
            return accuracy
        

    def train(self):
        train_running_loss = 0.0
        train_running_correct = 0

        # self.last_epoch = self.scheduler.last_epoch if self.scheduler is not None else self.last_epoch + 1
        self.last_epoch = self.scheduler.last_epoch + 1
        # self.optimizer.step()
        
        # if self.scheduler is not None:
        #     self.scheduler.step()

        pbar = tqdm(self.loader['train'])

        self.net.train()
        counter = 0
        
        curr_iter = self.last_epoch*len(self.loader)  
        all_iters = float(len(self.loader)*self.params['maximum_epochs'])  
        for batch_id, (inputs, labels) in enumerate(pbar):
            inputs, labels = self.prepare([inputs, labels]) # Prepare input and labels 
            counter+=labels.size(0)
            running_iter = curr_iter+batch_id+1
            outputs = self.net(inputs)
            
            #eden adds
            if type(self.net)==LANet: 
                outputs, aux = outputs 
                alpha = calc_alpha(running_iter, all_iters)

                main_loss = self.criterion(outputs, labels)
                aux_loss = self.criterion(aux, labels)
                loss = main_loss + alpha*aux_loss
            # loss = self.criterion(outputs, labels)
            # loss = CrossEntropy2d(outputs, labels, self.weight_cls[0]) # Calculate the loss function
            # loss = self.criterion(outputs, labels) 
            # dice_loss = DiceLoss(mode='multiclass', classes=self.weight_cls[0])(outputs, labels)
            # focal_loss = FocalLoss(mode='multiclass', gamma=2.0)(outputs, labels)
            # loss = dice_loss + (1*focal_loss)
            
            else:
                loss = self.criterion(outputs, labels)
             
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            # train_running_correct += (preds == labels).sum().item()
            train_running_correct += preds.eq(labels).sum().item()
            self.losses[self.iter_] = loss.item()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.mean_losses[self.iter_] = np.mean(self.losses[max(0,self.iter_-100):self.iter_])

            # clear()
            # print('Training (epoch {}/{}) [{}/{} ({:.0f}%)]\tIteração {}\tLoss: {:.4f}'.format(
            #         self.last_epoch, self.params['maximum_epochs'], batch_id, len(self.loader['train']),
            #         100. * batch_id / len(self.loader['train']), self.iter_, loss.item()
            #     )
            # )
            pbar.set_postfix({
                'Epoch': self.last_epoch, 
                'Batch': batch_id, 
                # 'Pixel-wise accuracy': sum(self.accuracies)/len(self.accuracies), 
                'Loss': loss.item(),
            })
            
            if self.iter_ % self.print_each == 0:
                image = inputs.data.cpu().numpy()[0]
                targets = labels.data.cpu().numpy()[0] #Labels
                # rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
                rgb = np.asarray(255 * np.transpose(image,(1,2,0)), dtype='uint8')
                pred = np.argmax(outputs.data.cpu().numpy()[0], axis=0)
                
                # acc = accuracy(pred, targets)
                
                # print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(epoch, self.params['maximum_epochs'], batch_idx, len(self.loader['train']),100. * batch_idx / len(self.loader['train']), loss.item(), accuracy(pred, gt)))
                fig = plt.figure()
                plt.plot(self.mean_losses[:self.iter_]) #and plt.show()
                fig.savefig(os.path.join(self.params['results_folder'], 'train_mean_loss'), dpi=fig.dpi, bbox_inches='tight')
                plt.close(fig)
                
                fig = plt.figure()
                ax1 = fig.add_subplot(131)
                ax1.set_title('RGB')
                plt.imshow(rgb)
                ax2 = fig.add_subplot(132)
                ax2.set_title('Ground truth')
                plt.imshow(convert_to_color(targets))
                ax3 = fig.add_subplot(133)
                ax3.set_title('Prediction')
                plt.imshow(convert_to_color(pred))
                # plt.show()
                fig.savefig(os.path.join(self.params['results_folder'], 'train_progress'), dpi=fig.dpi, bbox_inches='tight')
                plt.close(fig)
                
            self.iter_ += 1
            del(inputs, labels, loss)
        
        if self.scheduler is not None:
            self.scheduler.step()

        # Loss and accuracy for the complete epoch
        self.epoch_loss.append(train_running_loss/len(self.loader['train']))
        self.epoch_acc.append(100. * train_running_correct / counter)
        
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_loss), len(self.epoch_loss)).astype(int), self.epoch_loss, '-o')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train Loss/Epoch')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_loss'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)

        self.__save_plots()

    def __save_plots(self):
        # loss plots
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_loss), len(self.epoch_loss)).astype(int), self.epoch_loss, 'ro-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train Epoch/Loss')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_loss_curve'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)
        
        # acc plots
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_acc), len(self.epoch_acc)).astype(int), self.epoch_acc, 'bo-')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Train Epoch/Acc')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_acc_curve'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)
