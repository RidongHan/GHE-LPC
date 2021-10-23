import sklearn.metrics
import matplotlib
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

result_dir = './result/checkpoint/'

def main():
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 13,
    }
    models = sys.argv[1:]
    for model in models:
        x = np.load(os.path.join(result_dir, 'recall-' + model + '.npy')) 
        y = np.load(os.path.join(result_dir, 'precision-' + model + '.npy'))
        f1 = (2 * x * y / (x + y + 1e-20)).max()
        auc = sklearn.metrics.auc(x=x, y=y)
        #plt.plot(x, y, lw=2, label=model + '-auc='+str(auc))
        plt.plot(x, y, lw=1, label=model)
        print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1))
       
    plt.xlabel('Recall', font2)
    plt.ylabel('Precision', font2)
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall', font2)
    plt.legend(loc="upper right")
    labelss = plt.legend(loc='upper right').get_texts()
    [label.set_fontname('Times New Roman') for label in labelss]

    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'pr_curve'))
    foo_fig = plt.gcf()
    foo_fig.savefig(os.path.join(result_dir, 'pr_curve.eps'), format='eps')

if __name__ == "__main__":
    main()