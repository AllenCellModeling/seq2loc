import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import normalize as normalize_confmat
from torchvision.transforms.functional import to_tensor


def plot_confusion_matrix(test_y,
                          predict,
                          classes=np.array(["not mitotic",
                                            "M1: prophase 1",
                                            "M2: prophase 2",
                                            "M3: pro metaphase 1", 
                                            "M4: pro metaphase 2",
                                            "M5: metaphase",
                                            "M6: anaphase",
                                            "M7: telophase-cytokinesis"]),
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):

    
    cm = confusion_matrix(test_y, predict)
    cm_normalized = normalize_confmat(cm, axis=1, norm='l1')

    fig, ax = plt.subplots(figsize=(8,8), dpi=100)    
    cax = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    for i,j in itertools.product(*[range(d) for d in cm.shape]):
        ax.text(j, i, cm[i,j], horizontalalignment="center", color="white" if cm_normalized[i, j] > 0.5 else "black")

    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.grid(b=False)
    
    return(fig, ax)
    
def print_confusion_matrix(test_y,
                           predict,
                           classes=np.array(["not mitotic",
                                             "M1: prophase 1",
                                             "M2: prophase 2",
                                             "M3: pro metaphase 1", 
                                             "M4: pro metaphase 2",
                                             "M5: metaphase",
                                             "M6: anaphase",
                                             "M7: telophase-cytokinesis"])):
    
    plot_confusion_matrix(test_y, predict, classes=classes)
    plt.show()

def print_numerical_report(test_y,
                           predict,
                           classes=np.array(["not mitotic",
                                             "M1: prophase 1",
                                             "M2: prophase 2",
                                             "M3: pro metaphase 1", 
                                             "M4: pro metaphase 2",
                                             "M5: metaphase",
                                             "M6: anaphase",
                                             "M7: telophase-cytokinesis"])):
    
    print(classification_report(test_y, predict))
    print('accuracy', accuracy_score(test_y, predict))

    
def model_analysis(test_y,
                   predict,
                   classes=np.array(["not mitotic",
                                     "M1: prophase 1",
                                     "M2: prophase 2",
                                     "M3: pro metaphase 1", 
                                     "M4: pro metaphase 2",
                                     "M5: metaphase",
                                     "M6: anaphase",
                                     "M7: telophase-cytokinesis"])):
    
    print_numerical_report(test_y, predict, classes=classes)
    print_confusion_matrix(test_y, predict, classes=classes)
    
def fig_to_torch(myfig):
    myfig.canvas.draw();    
    return to_tensor(np.array(myfig.canvas.renderer._renderer)[:,:,:3])

def torch_confusion_matrix(test_y,
                           predict,
                           classes=np.array(["not mitotic",
                                             "M1: prophase 1",
                                             "M2: prophase 2",
                                             "M3: pro metaphase 1", 
                                             "M4: pro metaphase 2",
                                             "M5: metaphase",
                                             "M6: anaphase",
                                             "M7: telophase-cytokinesis"])):

    fig, ax = plot_confusion_matrix(test_y, predict, classes=classes)
    fig.canvas.draw();
    out = np.array(fig.canvas.renderer._renderer)[:,:,:3]
    plt.close(fig)
    
    return to_tensor(out)
