import numpy as np
import os
import time
import sys
sys.path.insert(0, './model')
from model.GraphToProperty import Graph2Property
import tensorflow as tf
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import pickle


def loadInputs(FLAGS, idx, modelName, unitLen):
    adj = None
    features = None
    adj = np.load('./database/' + FLAGS.database + '/adj/' + str(idx) + '.npy')
    features = np.load('./database/' + FLAGS.database + '/features/' + str(idx) + '.npy')
    retInput = (adj, features)
    retOutput = (np.load('./database/' + FLAGS.database + '/' + FLAGS.output + '.npy')).astype(float)
    # [idx * unitLen:(idx + 1) * unitLen]).astype(float)
    return retInput, retOutput


def training(model, FLAGS, modelName):
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    decay_rate = FLAGS.decay_rate
    save_every = FLAGS.save_every
    learning_rate = FLAGS.learning_rate
    num_DB = FLAGS.num_DB
    unitLen = FLAGS.unitLen
    total_iter = 0
    total_st = time.time()
    print("Start Training XD")
    for epoch in range(num_epochs):
        # Learning rate scheduling
        model.assign_lr(learning_rate * (decay_rate ** epoch))
        for i in range(0, 3):
            _graph, _property = loadInputs(FLAGS, i, modelName, unitLen)
            num_batches = int(_graph[0].shape[0] / batch_size)
            st = time.time()
            for _iter in range(num_batches):
                total_iter += 1
                A_batch = _graph[0][_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                X_batch = _graph[1][_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                P_batch = _property[_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                if total_iter % 5 != 0:
                    # Training
                    cost = model.train(A_batch, X_batch, P_batch)
                    print("train_iter:", total_iter, ",epoch:", epoch, ",cost:", cost)
                    # 将训练时的损失函数值记录
                    train_loss_list.append(cost)
                elif total_iter % 5 == 0:
                    # Test accuracy
                    Y, cost = model.test(A_batch, X_batch, P_batch)
                    print("test_iter : ", total_iter, ", epoch : ", epoch, ", cost :  ", cost)
                    if (total_iter % 100 == 0):
                        mse = (np.mean(np.power((Y.flatten() - P_batch), 2)))
                        mae = (np.mean(np.abs(Y.flatten() - P_batch)))
                        # 记录测试集的误差
                        mse_list.append(mse)
                        mae_list.append(mae)
                        print("MSE : ", mse, "\t MAE : ", mae)
                if total_iter % save_every == 0:
                    # Save network!
                    ckpt_path = 'save/' + modelName + '.ckpt'
                    model.save(ckpt_path, total_iter)
            et = time.time()
            print("time : ", et - st)
            st = time.time()
    total_et = time.time()
    print("Finish training! Total required time for training : ", (total_et - total_st))
    return


# execution : ex)  python train.py GCN logP 3 100 0.001 0.95
# method = sys.argv[1]
# prop = sys.argv[2]
# num_layer = int(sys.argv[3])
# epoch_size = int(sys.argv[4])
# learning_rate = float(sys.argv[5])
# decay_rate = float(sys.argv[6])

method = "GCN"
prop = "logP"
num_layer = 3
epoch_size = 100
learning_rate = 0.001
decay_rate = 0.95

database = ''
if (prop in ['TPSA2', 'logP', 'SAS']):
    database = 'ZINC'
    numDB = 45
    unit_len = 10000
elif (prop == 'pve'):
    database = 'CEP'
    numDB = 27
    unit_len = 1000
print('method :', method, '\t prop :', prop, '\t num_layer :', num_layer, '\t epoch_size :', epoch_size,
      '\t learning_rate :', learning_rate, '\t decay_rate :', decay_rate, '\t Database :', database, '\t num_DB :',
      numDB)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set FLAGS for environment setting
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', method, 'GCN, GCN+a, GCN+g, GCN+a+g')
flags.DEFINE_string('output', prop, '')
flags.DEFINE_string('loss_type', 'MSE', 'Options : MSE, CrossEntropy, Hinge')  ### Using MSE
flags.DEFINE_string('database', database, 'Options : ZINC, CEP')  ### Using MSEr
flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp')
flags.DEFINE_string('readout', 'atomwise', 'Options : atomwise, graph_gather')
flags.DEFINE_integer('latent_dim', 512, 'Dimension of a latent vector for autoencoder')
flags.DEFINE_integer('num_layers', num_layer, '# of hidden layers')
flags.DEFINE_integer('epoch_size', epoch_size, 'Epoch size')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('save_every', 1000, 'Save every')
flags.DEFINE_float('learning_rate', learning_rate, 'Batch size')
flags.DEFINE_float('decay_rate', decay_rate, 'Batch size')
flags.DEFINE_integer('num_DB', numDB, '')
flags.DEFINE_integer('unitLen', unit_len, '')
modelName = FLAGS.model + '_' + str(FLAGS.num_layers) + '_' + FLAGS.output + '_' + FLAGS.readout + '_' + str(
    FLAGS.latent_dim) + '_' + FLAGS.database
print("Summary of this training & testing")
print("Model name is", modelName)
print("A Latent vector dimension is", str(FLAGS.latent_dim))
print("Using readout funciton of", FLAGS.readout)
print("A learning rate is", str(FLAGS.learning_rate), "with a decay rate", str(FLAGS.decay_rate))
print("Using", FLAGS.loss_type, "for loss function in an optimization")
print("num_DB is", numDB)
# 存储误差列表
train_loss_list = []
mse_list = []
mae_list = []
model = Graph2Property(FLAGS)
training(model, FLAGS, modelName)

# 绘制损失函数变化
x1 = np.arange(FLAGS.epoch_size)
plt.plot(train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 9)
plt.show()
# 绘制准确率变化
# x = np.arange(FLAGS.epoch_size)
# plt.plot(x, mse_list, label='MSE')
# plt.plot(x, mae_list, label='MAE', marker='s')
# plt.xlabel("epochs")
# plt.ylabel("error")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()

