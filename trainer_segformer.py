from pathlib import Path
from torch.autograd import Variable
import torch
import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
# from datasets import load_metric
import evaluate
from torch import nn
from sklearn.metrics import accuracy_score

from utils import clear, count_sliding_window, make_optimizer, make_scheduler, CrossEntropy2d, accuracy, metrics, save_test, sliding_window, grouper, convert_from_color, convert_to_color, global_accuracy

class TrainerSegformer():
    
    def __init__(self, net, loader, params, scheduler = True, cbkp = None,):
        
        self.net = net
        self.loader = loader
        self.params = params
        self.iter_ = 0
        self.epoch_loss = []
        self.accuracies = []
        self.losses = []
        self.mean_losses = np.zeros(100000000)
        self.metric = evaluate.load("mean_iou")
        self.print_each = params['print_each'] or 100 # Print statistics every 500 iterations
        
        # # Weights for class balancing
        self.weight_cls = self.prepare([self.params['weights']])
        # self.criterion = FocalLoss(weight=self.weight_cls[0], gamma=3.0)
        
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

    def compute_metrics(self, eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = self.metric._compute(
                    predictions=pred_labels,
                    references=labels,
                    num_labels=len(self.params['id2label']),
                    ignore_index=0,
                    reduce_labels=False,
                )
            
            # add per category metrics as individual key-value pairs
            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update({f"accuracy_{self.params['id2label'][i]}": v for i, v in enumerate(per_category_accuracy)})
            metrics.update({f"iou_{self.params['id2label'][i]}": v for i, v in enumerate(per_category_iou)})

            return metrics
            
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
            path = './{}_model_final.pth.tar'.format(self.model_id)

        # Save current loss, epoch, model weights and optimizer params
        torch.save({
            'epoch': self.last_epoch,
            'losses': self.losses,
            'mean_losses': self.mean_losses,
            'epoch_loss': self.epoch_loss,
            'accuracies': self.accuracies,
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
            Path(f'./tmp/{filepath}').mkdir(parents=True, exist_ok=True)
            
            pred = np.zeros(img.shape[:2] + (self.params['n_classes'],))

            total = count_sliding_window(img, step=test_stride, window_size=test_ws) // bs
            for i, coords in enumerate(tqdm(grouper(bs, sliding_window(img, step=test_stride, window_size=test_ws)), total=total, leave=False)):
                # Display in progress results
                if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1) #Usar argmax do pytorch
                    fig = plt.figure()
                    fig.add_subplot(1,3,1)
                    plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    fig.add_subplot(1,3,2)
                    plt.imshow(convert_to_color(_pred))
                    fig.add_subplot(1,3,3)
                    plt.imshow(gt)
                    # plt.show()
                    fig.savefig(f"./tmp/test_progress", dpi=fig.dpi, bbox_inches='tight')
                    plt.clf()
                        
                # Build the tensor (ex: 2048px/256px = 8 patches de 256x256)
                with torch.no_grad():
                    image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = Variable(torch.from_numpy(image_patches).cuda())
                
                # Do the inference
                outs = self.net(image_patches)
                loss, logits = outs.loss, outs.logits
                # outs = outs.detach().cpu().numpy()
                upsampled_logits = nn.functional.interpolate(
                    logits, 
                    size=image_patches.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1)
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(upsampled_logits, coords):
                    pred[x:x+w, y:y+h] += out.detach().cpu().numpy().transpose(1,2,0)
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
            fig.savefig(f"./tmp/{filepath}/result", dpi=fig.dpi, bbox_inches='tight')
            plt.clf()

            all_preds.append(pred)
            all_gts.append(gt_e)

            # clear()
            # Compute some metrics
            
            metrics(pred.ravel(), gt_e.ravel(), label_values=self.params['classes'], filepath=filepath)
            accuracy = metrics(
                predictions=np.concatenate([p.ravel() for p in all_preds]), 
                gts=np.concatenate([p.ravel() for p in all_gts]).ravel(),
                label_values=self.params['classes'],
                all=True,
                filepath=filepath
            )

        if all:
            save_test(acc=accuracy, all_preds=all_preds, all_gts=all_gts)
            return accuracy, all_preds, all_gts
        else:
            return accuracy
        

    def train(self):
        running_loss = 0.0
        # self.last_epoch = self.scheduler.last_epoch if self.scheduler is not None else self.last_epoch + 1
        self.last_epoch = self.scheduler.last_epoch + 1
        # self.optimizer.step()
        
        # if self.scheduler is not None:
        #     self.scheduler.step()

        pbar = tqdm(self.loader['train'])

        self.net.train()
        for batch_id, (inputs, labels) in enumerate(pbar):
            inputs, labels = self.prepare([inputs, labels]) # Prepare input and labels 

            # zero the parameter gradients
            self.optimizer.zero_grad()
            
            outputs = self.net(pixel_values=inputs, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            running_loss += loss.item()
            
            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)
                mask = (labels != 255) # we don't include the background class in the accuracy calculation
                pred_labels = predicted[mask].detach().cpu().numpy()
                true_labels = labels[mask].detach().cpu().numpy()

                accuracy = accuracy_score(pred_labels, true_labels)
                # loss_ = CrossEntropy2d(upsampled_logits, labels, self.weight_cls[0]) # Calculate the loss function
                self.accuracies.append(accuracy)
                # self.losses[self.iter_] = loss.item()
                self.losses.append(loss.item())
                self.mean_losses[self.iter_] = np.mean(self.losses[max(0,self.iter_-100):self.iter_])
                self.metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

                pbar.set_postfix({
                    'Epoch': self.last_epoch, 
                    'Batch': batch_id, 
                    'Pixel-wise accuracy': sum(self.accuracies)/len(self.accuracies), 
                    'Loss': sum(self.losses)/len(self.losses)}
                )
            
            # print('Training (epoch {}/{}) [{}/{} ({:.0f}%)]\tIteração {}\tLoss: {:.4f}'.format(
            #         self.last_epoch, self.params['maximum_epochs'], batch_id, len(self.loader['train']),
            #         100. * batch_id / len(self.loader['train']), self.iter_, loss.item()
            #     )
            # )
            
            if self.iter_ % self.print_each == 0:
                metrics = self.metric.compute(num_labels=len(self.params['id2label']), 
                                    ignore_index=255,
                                    reduce_labels=False, # we've already reduced the labels before)
                )

                # print(f"Mean IoU: {metrics['mean_iou']}\tMean Acc: {metrics['mean_accuracy']}")
                pbar.set_postfix({
                    'Epoch': self.last_epoch, 
                    'Batch': batch_id, 
                    'Pixel-wise accuracy': sum(self.accuracies)/len(self.accuracies), 
                    'Loss': sum(self.losses)/len(self.losses),
                    'Mean IoU': metrics['mean_iou'],
                    'Mean Acc': metrics['mean_accuracy']
                })

                image = inputs.detach().cpu().numpy()[0] #Labels
                image = np.asarray(255 * np.transpose(image,(1,2,0)), dtype='uint8')

                gt = labels.detach().cpu().numpy()[0]

                predicted_ = convert_to_color(predicted.detach().cpu().numpy()[0])

                # Show image + mask
                image_mask = np.array(image) * 0.5 + predicted_ * 0.5
                image_mask = image_mask.astype(np.uint8)
                
                fig = plt.figure()
                plt.plot(self.mean_losses[:self.iter_]) #and plt.show()
                fig.savefig(f"./tmp/train_mean_loss", dpi=fig.dpi, bbox_inches='tight')
                plt.clf()
                
                fig = plt.figure()
                ax1 = fig.add_subplot(141)
                ax1.set_title('RGB')
                plt.imshow(image)
                ax2 = fig.add_subplot(142)
                ax2.set_title('Ground truth')
                plt.imshow(convert_to_color(gt))
                ax3 = fig.add_subplot(143)
                ax3.set_title('Prediction')
                plt.imshow(predicted_)
                ax4 = fig.add_subplot(144)
                ax4.set_title('GTxPred')
                plt.imshow(image_mask)
                # plt.show()
                fig.savefig(f"./tmp/train_progress", dpi=fig.dpi, bbox_inches='tight')
                plt.clf()
                
            self.iter_ += 1
            del(inputs, labels, loss)
        
        if self.scheduler is not None:
            self.scheduler.step()

        self.epoch_loss.append(running_loss/len(self.loader['train']))
        
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_loss), len(self.epoch_loss)).astype(int), self.epoch_loss, '-o')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train Loss/Epoch')
        fig.savefig(f"./tmp/train_epoch_loss", dpi=fig.dpi, bbox_inches='tight')
        plt.clf()


    