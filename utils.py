# Utils
import random, os, copy
import numpy as np
import torch
import itertools
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
from os import system, name

#### TRAINER UTILS ####
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim as optim
# import segmentation_models_pytorch as smp

# palette = {
#     0 : (255, 0, 0),        # Desenvolvimento (vermelho)
#     1 : (38, 115, 0),       # Floresta Mata (verde escuro)
#     2 : (0, 255, 197),      # Piscina (ciano)
#     3 : (0, 0, 0),          # Sombra (preto)
#     4 : (133, 199, 126),    # Floresta Regeneração (verde claro)
#     5 : (255, 255, 0),      # Agricultura (amarelo)
#     6 : (255, 85, 0),       # Formação Rochosa (laranja)
#     7 : (115, 76, 0),       # Solo Exposto (marrom)
#     8 : (84, 117, 168),     # Água (azul escuro)
# }
palette = {
    0 : (255, 0, 0),        # Desenvolvimento (vermelho)
    1 : (38, 115, 0),       # Floresta Mata (verde escuro)
    # 2 : (0, 255, 197),      # Piscina (ciano)
    2 : (0, 0, 0),          # Sombra (preto)
    3 : (133, 199, 126),    # Floresta Regeneração (verde claro)
    4 : (255, 255, 0),      # Agricultura (amarelo)
    5 : (255, 85, 0),       # Formação Rochosa (laranja)
    6 : (115, 76, 0),       # Solo Exposto (marrom)
    7 : (84, 117, 168),     # Água (azul escuro)
}

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    # W, H = img.shape[:-1] # CHANGED
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def CrossEntropy2d(input, target, weight=None, reduction='mean'):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, reduction=reduction)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, reduction=reduction)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
        
"""  Make optimizer routine"""
def make_optimizer(args, net):
    trainable = filter(lambda x: x.requires_grad, net.parameters()) # Only the parameters that requires gradient are passed to the optimizer

    if args['optimizer'] == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args['momentum']}
    elif args['optimizer'] == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon']
        }
    elif args['optimizer'] == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args['epsilon']}

    kwargs['lr'] = args['lr']
    kwargs['weight_decay'] = args['weight_decay']
    
    return optimizer_function(trainable, **kwargs)



""" Make scheduler routine """
def make_scheduler(args, optimizer):
    if args['type'] == 'multi':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=args['milestones'],
            gamma=args['gamma']
        )
    else:
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args['lr_decay'],
            gamma=args['gamma']
        )
    return scheduler


def calculate_cm(predictions, labels, label_values = None, normalize = None):
    return confusion_matrix(labels, predictions, labels=label_values, normalize=normalize)


""" Global acurracy metric calculation """
def global_accuracy(predictions, labels):
    # Calculate confusion matrix
    cm = calculate_cm(predictions, labels)
    # Sum all values in main diagonal
    main_diagonal = sum([cm[i][i] for i in range(len(cm))])
    # return TP+TN / TP+TN+FN x 100%
    return 100 * float(main_diagonal) / np.sum(cm)

def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    # dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image.cpu().numpy().transpose(1,2,0))
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def metrics(predictions, gts, label_values, all=False, filepath=None):
    
    txt = []
    cm = confusion_matrix(
            gts,
            predictions,
            labels=range(len(label_values)))
    
    # plt.show()
    # plot_confusion_matrix(
    #     cm           = cm, 
    #     normalize    = False,
    #     target_names = range(len(label_values)),
    #     title        = "Confusion Matrix"
    # )
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=label_values)
    fig.savefig(f"./tmp/cm_all" if all else f'./tmp/{filepath}/cm' , dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)
    
    print("Confusion matrix :")
    print(cm)
    txt.append("Confusion Matrix")
    txt.append(cm)
    
    print("---")
    txt.append("------")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    txt.append("{} pixels processed".format(total))
    txt.append("Total accuracy : {}%".format(accuracy))
    
    print("---")
    txt.append("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    txt.append('\nF1Score :')
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))
        txt.append("{}: {}".format(label_values[l_id], score))

    print("---")
    txt.append("---")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    txt.append("\nKappa: " + str(kappa))
    
    return accuracy

# define our clear function
def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
        
def plot_confusion_matrix_local(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    # plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save:
        fig.savefig(f"./tmp/cm", dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean, std

def save_test(acc, all_preds, all_gts, path=None):
    try:
        print('************** save test results **************')
        if path is None:
            path = './segnet256_test_result.npz'
            
        np.savez_compressed(path, {
            'acc': acc,
            'all_preds': all_preds,
            'all_gts': all_gts,
        })
    except:
        print('[AVISO] Erro ao salvar os resultados do teste!')
        pass
    
def load_test(path=None):
    assert os.path.exists(path), "{} cant be opened".format(path)
    
    data = np.load(path, allow_pickle=True)
    return data.item()

def save_loss_weights(data, path=None):
    if path is None:
        path = './loss_weights.npy'
    np.save(path, data)
        
def load_loss_weights(path=None):
    if path is None:
        path = './loss_weights.npy'
    x = np.load(path, allow_pickle=True)
    return x.item()

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# import torchvision.models as models
# import torch.nn as nn
# def build_model_efficientnet_b0(num_classes, pretrained=True, fine_tune=True):
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights')
#     else:
#         print('[INFO]: Not loading pre-trained weights')
#     model = models.efficientnet_b0(pretrained=pretrained)
#     if fine_tune:
#         print('[INFO]: Fine-tuning all layers...')
#         for params in model.parameters():
#             params.requires_grad = True
#     elif not fine_tune:
#         print('[INFO]: Freezing hidden layers...')
#         for params in model.parameters():
#             params.requires_grad = False
#     # Change the final classification head.
#     model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
#     return model
