import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import pandas as pd
import SNN
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools


'''
Define the plot confusion matrix

'''


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.autoscale()


'''
Model Test

'''

input = tf.placeholder(tf.float32)
input_exp = tf.exp(input)
groundtruth = tf.placeholder(tf.float32)

try:
    w1 = np.load('weight_nslkdd11.npy')
    w2 = np.load('weight_nslkdd12.npy')
    w3 = np.load('weight_nslkdd13.npy')
    layer_in = SNN.SNNLayer(312, 100, w1)
    layer_out1 = SNN.SNNLayer(100, 100, w2)
    layer_out2 = SNN.SNNLayer(100, 5, w3)
    print('Weight loaded!')
except:
    layer_in = SNN.SNNLayer(312, 100)
    layer_out1 = SNN.SNNLayer(100, 100)
    layer_out2 = SNN.SNNLayer(100, 5)
    print('No weight file found, use random weight')

layerin_out = layer_in.forward(input_exp)
layerout_out1 = layer_out1.forward(layerin_out)
layerout_out2 = layer_out2.forward(layerout_out1)

# the time of output
nnout = tf.log(layerout_out2)


config = tf.ConfigProto(device_count={'GPU': 0})
config.gpu_options.allow_growth = True
sess = tf.Session()
sess.run(tf.global_variables_initializer())


BATCH_SIZE = 10

scale = 3
mnist = SNN.Nslkdd(path=["dataset1/X_test.npy","dataset1/y_test.npy"])



print('testing started')
step = 1

accuracy = []

y_pred = None
y_test = None

while(True):
    xs, ys = mnist.next_batch(BATCH_SIZE, shuffle=False)
    xs = scale * xs  # change the concave seal to convex seal for feature extraction.
    out = sess.run(nnout, {input: xs, groundtruth: ys})
    ys_1 = np.argmax(ys, axis=1)

    if y_pred is None:
        y_test = ys_1
    else:
        y_test = np.append(y_test, ys_1, axis=0)

    layerout = np.argmin(out, axis=1)
    # print(layerout)
    if y_pred is None:
        y_pred = layerout
    else:
        y_pred = np.append(y_pred, layerout, axis=0)
    # y_pred.append(layerout)
    # print(ys_1)
    # print(y_pred)
    layerout = to_categorical(layerout, 5)


    # print(layerout)
    # print(ys)
    accurate = 0
    for i in range(len(ys)):
        if (layerout[i] == ys[i]).all():
            accurate = accurate + 1
    accurate = accurate / 10.
    # print(accurate)
    accuracy.append(accurate)
    if step % 10 == 0:
        accurate1 = tf.reduce_mean(accuracy)
        print('Step: ' + repr(step) + ', ' + 'Accurate: ' + repr(sess.run(accurate1)))
    step = step + 1
    if step == 2254:
        break


'''

Calculate the Accuracy, Precision, F1 and Recall. 
Result visualization.

'''

print('\r\nLabel: ' + str(y_test[:20]))
print('Prediction: ' + str(y_pred[:20]))

actions = y_pred
maped = y_test
attack_types = ['normal', 'Dos', 'Probe', 'R2L', 'U2R']


# Result visualization 1
total_reward = 0
true_labels = np.zeros(len(attack_types),dtype=int)
estimated_labels = np.zeros(len(attack_types),dtype=int)
estimated_correct_labels = np.zeros(len(attack_types),dtype=int)

labels,counts = np.unique(maped, return_counts=True)
true_labels[labels] += counts
for indx, a in enumerate(actions):
    estimated_labels[a] += 1
    if a == maped[indx]:
        total_reward += 1
        estimated_correct_labels[a] += 1

action_dummies = pd.get_dummies(actions)
posible_actions = np.arange(len(attack_types))
for non_existing_action in posible_actions:
    if non_existing_action not in action_dummies.columns:
        action_dummies[non_existing_action] = np.uint8(0)
labels_dummies = pd.get_dummies(maped)

normal_f1_score = f1_score(labels_dummies[0].values, action_dummies[0].values)
dos_f1_score = f1_score(labels_dummies[1].values, action_dummies[1].values)
probe_f1_score = f1_score(labels_dummies[2].values, action_dummies[2].values)
r2l_f1_score = f1_score(labels_dummies[3].values, action_dummies[3].values)
u2r_f1_score = f1_score(labels_dummies[4].values, action_dummies[4].values)

Accuracy = [normal_f1_score, dos_f1_score, probe_f1_score, r2l_f1_score, u2r_f1_score]
Mismatch = estimated_labels - true_labels

acc = float(100 * total_reward / len(maped))
print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {:.2f}%'.format(total_reward,
                                                                                 len(maped), acc))
outputs_df = pd.DataFrame(index=attack_types, columns=["Estimated", "Correct", "Total", "F1_score"])
for indx, att in enumerate(attack_types):
    outputs_df.iloc[indx].Estimated = estimated_labels[indx]
    outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
    outputs_df.iloc[indx].Total = true_labels[indx]
    outputs_df.iloc[indx].F1_score = Accuracy[indx] * 100
    outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])

print(outputs_df)


# Result visualization 2
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
width = 0.35
pos = np.arange(len(true_labels))
p1 = plt.bar(pos, estimated_correct_labels,width,color='g')
p2 = plt.bar(pos+width,
             (np.abs(estimated_correct_labels-true_labels)),width,
             color='r')
p3 = plt.bar(pos+width,np.abs(estimated_labels-estimated_correct_labels),width,
             bottom=(np.abs(estimated_correct_labels-true_labels)),
             color='b')

ax.yaxis.set_tick_params(labelsize=15)
ax.set_xticks(pos+width/2)
ax.set_xticklabels(attack_types,rotation='vertical',fontsize = 'xx-large')

plt.legend(('Correct estimated','False negative','False positive'),fontsize = 'x-large')
plt.tight_layout()
plt.savefig('results/test_adv_imp.svg', format='svg', dpi=1000)


# Result visualization 3
aggregated_data_test = np.array(maped)

print('\r\nPerformance measures on Test data')
print('Accuracy =  {:.4f}'.format(accuracy_score( aggregated_data_test,actions)))
print('F1 =  {:.4f}'.format(f1_score(aggregated_data_test,actions, average='weighted')))
print('Precision_score =  {:.4f}'.format(precision_score(aggregated_data_test,actions, average='weighted')))
print('recall_score =  {:.4f}'.format(recall_score(aggregated_data_test,actions, average='weighted')))

cnf_matrix = confusion_matrix(aggregated_data_test,actions)
np.set_printoptions(precision=2)
plt.figure()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('results/confusion_matrix_adversarial.svg', format='svg', dpi=1000)

plt.figure()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=attack_types,
                      title='Confusion matrix, raw number')
plt.savefig('results/confusion_matrix_raw_number.svg', format='svg', dpi=1000)

#plt.show()
# classification report
print(classification_report(aggregated_data_test,actions,target_names=['normal', 'Dos', 'Probe', 'R2L', 'U2R']))


# Result visualization 4
mapa = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
yt_app = pd.Series(maped).map(mapa)

perf_per_class = pd.DataFrame(index=range(len(yt_app.unique())), columns=['name', 'acc', 'f1', 'pre', 'rec'])
for i, x in enumerate(pd.Series(yt_app).value_counts().index):
    y_test_hat_check = pd.Series(actions).map(mapa).copy()
    y_test_hat_check[y_test_hat_check != x] = 'OTHER'
    yt_app = pd.Series(maped).map(mapa).copy()
    yt_app[yt_app != x] = 'OTHER'
    ac = accuracy_score(yt_app, y_test_hat_check)
    f1 = f1_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
    pr = precision_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
    re = recall_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
    perf_per_class.iloc[i] = [x, ac, f1, pr, re]

print("\r\nOne vs All metrics: \r\n{}".format(perf_per_class))

