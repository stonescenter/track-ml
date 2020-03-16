__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import os
import numpy as np
from math import sqrt

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



def evaluate_model(history, model, x, y, save_to):

    eval_model=model.evaluate(x, y)

    print('loss: {:4f}'.format(eval_model[0]))
    print('accuracy: {:4f}'.format(eval_model[1]))
    print(history)
    plt.plot(history.history['loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(save_to)
    plt.show()


def evaluate_training(history, save_to):
    history = history.history

    print(history.keys())
    print('[Ouput] Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    save_fname = os.path.join(save_to, 'eval-accuracy.png') 
    plt.savefig(save_fname)
    plt.show()
    # summarize history for loss
    plt.clf()
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    save_fname = os.path.join(save_to, 'eval-loss.png') 
    plt.savefig(save_fname)
    plt.show()    

    print('[Output] resulta saved at %s', save_fname)

def calc_score(y_true, y_predicted, report=False):

    r2 = r2_score(y_true, y_predicted)
    rmse = sqrt(mean_squared_error(y_true, y_predicted))
    mae = mean_absolute_error(y_true, y_predicted)

    report_string = "[Output] " 
    report_string += "---Regression Scores--- \n"
    report_string += "\tR_2 statistics        (R2)  = " + str(round(r2,3)) + "\n"
    report_string += "\tRoot Mean Square Error(RMSE) = " + str(round(rmse,3)) + "\n"
    report_string += "\tMean Absolute Error   (MAE) = " + str(round(mae,3)) + "\n"

    if report:
        print(report_string)

    return mae, report_string

def evaluate_forecast(y_true, y_predicted):
    '''
        Return 
            score  : return the score total with RMSE
            scores : return the score RMSE for each features
    '''
    rmses = []
    r2s = []
    y_true = np.array(y_true)
    
    rows = y_true.shape[0]
    cols = y_true.shape[1]

    for i in range(cols):
        mse = mean_squared_error(y_true[:,i], y_predicted[:,i])
        rmses.append(sqrt(mse))
        
        r2 = r2_score(y_true[:,i], y_predicted[:,i])
        r2s.append(r2)
        
    # calculate total see definition of MSE
    s = 0
    for row in range(rows):
        for col in range(cols):
            s += (y_true[row, col] - y_predicted[row,col])**2
        
    rmst = sqrt(s/(cols*rows))

    return r2s, rmst, rmses

def summarize_scores(r2, score, scores):
    s_scores = ', '.join(['%.2f' % s for s in scores])
    s_r2 = ', '.join(['%.2f' % s for s in r2])
    #print('RMSE:\t\t[%.3f] \nRMSE features: \t[%s] \nR^2  features:\t[%s] ' % (score, s_scores, s_r2))
    print('R^2  features:\t[%s] \nRMSE:\t\t[%.3f] \nRMSE vector: \t[%s] ' % (s_r2, score, s_scores))