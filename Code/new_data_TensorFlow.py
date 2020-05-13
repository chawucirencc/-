#!/usr/bin/env.python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib import layers # 在TensorFlow2.X版本中去掉了tensorflow.contrib


def load_data(file_path):
    file_data = pd.read_excel(file_path)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_columns', 20)
    # print(file_data.head())
    # print(file_data.info())
    return file_data


def plot_feature(file_data):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    # Part 1
    fig = plt.figure()
    fig.set(alpha=0.8)
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    fig.add_subplot(221)
    file_data['Exited'].value_counts().plot(kind='bar', width=0.3)
    plt.title('流失客户与未流失客户'); plt.xlabel('1为流失客户'); plt.ylabel('人数')
    fig.add_subplot(222)
    file_data['Gender'].value_counts().plot(kind='bar', width=0.3)
    plt.title('性别情况'); plt.ylabel('人数')
    fig.add_subplot(223)
    file_data['Geography'].value_counts().plot(kind='bar')
    plt.title('用户所在国家情况'); plt.ylabel('人数')
    fig.add_subplot(224)
    file_data['Age'].plot(kind='kde', color='g')
    plt.title('年龄频率图'); plt.xlabel('年龄')
    # Part 2
    gender_f = file_data['Exited'][file_data['Gender'] == 'Female'].value_counts()
    gender_m = file_data['Exited'][file_data['Gender'] == 'Male'].value_counts()
    gender_Df = pd.DataFrame({'Female': gender_f, 'Male':gender_m})
    gender_Df.plot(kind='bar', stacked=True)
    plt.title('流失客户和性别的关系'); plt.ylabel('人数')

    Geo_f = file_data['Exited'][file_data['Geography'] == 'France'].value_counts()
    Geo_s = file_data['Exited'][file_data['Geography'] == 'Spain'].value_counts()
    Geo_g = file_data['Exited'][file_data['Geography'] == 'Germany'].value_counts()
    Geo_df = pd.DataFrame({'France': Geo_f, 'Spain':Geo_s, 'Germany': Geo_g})
    Geo_df.plot(kind='bar', stacked=True)
    plt.title('流失客户和用户所在国家的关系'); plt.ylabel('人数')
    plt.show()


def feature_processing(file_data):
    std = StandardScaler()
    def change_gender(gen):
        if gen == 'Female':
            return 0
        elif gen == 'Male':
            return 1
    file_data['Gender'] = file_data['Gender'].apply(change_gender)

    def change_tenure(ten):
        if ten < 6:
            return 0
        else:
            return 1
    # file_data['Tenure'] = file_data['Tenure'].apply(change_tenure)
    def change_Num(n) :
        if n == 1:
            return 0
        else:
            return 1
    file_data['NumOfProducts'] = file_data['NumOfProducts'].apply(change_Num)

    file_data['CreditScore'] = std.fit_transform(np.array(file_data['CreditScore']).reshape((-1, 1)))
    file_data['Tenure'] = np.sqrt(np.array(file_data['Tenure']).reshape((-1, 1)))
    # file_data['Age'] = np.sqrt(np.array(file_data['Age']).reshape((-1, 1)))
    file_data['Age'] = std.fit_transform(np.array(file_data['Age']).reshape((-1, 1)))
    file_data['Balance'] = std.fit_transform(np.array(file_data['Balance']).reshape((-1, 1)))
    file_data['EstimatedSalary'] = std.fit_transform(np.array(file_data['EstimatedSalary']).reshape((-1, 1)))

    Geography = pd.get_dummies(file_data['Geography'], prefix='Geography')
    data = pd.concat([file_data, Geography], axis=1)
    to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Geography_Spain']
    data = data.drop(to_drop, axis=1)
    print(data.head())
    print(data.columns)
    return data


def plot_feature_score(data):
    """
    画出特征评分图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    feature = data.columns
    select = SelectKBest(f_classif, k=5)
    select.fit(data[feature], data['Exited'])
    score = -np.log10(select.pvalues_)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(feature)), score)
    plt.title('各个特征评分'); plt.ylabel('评分')
    plt.xticks(range(len(feature)), feature, rotation=90)
    plt.show()


def split_data(data):
    y = np.array(data['Exited']).reshape((-1, 1))
    data = data.drop('Exited', axis=1)
    X = data.values
    one = OneHotEncoder()
    y_ = one.fit_transform(y).toarray()
    return X, y_


def stand_data(X):
    std = StandardScaler()
    X_std = std.fit_transform(X)
    return X_std


def create_model(X, y):
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    train_data, ver_data, train_label, ver_label = train_test_split(X_train_all, y_train_all,
                                                                    test_size=0.2, random_state=1)
    bath_size = 100  # 批量大小

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 11], name='x_input')
        y = tf.placeholder(tf.float32, [None, 2], name='y_input')
        keep_prob = tf.placeholder(tf.float32)
    # 输入层
    with tf.name_scope('input_layer'):
        with tf.name_scope('Weight'):
            W_input = tf.Variable(tf.truncated_normal([11, 150], stddev=0.1, dtype=tf.float32), name='weight1')
            variable_summaries(W_input)
        with tf.name_scope('biases'):
            b_input = tf.Variable(tf.zeros([150])+0.1, dtype=tf.float32, name='biases1')
            variable_summaries(b_input)
        with tf.name_scope('W_mat_b'):
            W_mat_b = tf.matmul(x, W_input) + b_input
        with tf.name_scope('L1'):
            L1 = tf.nn.tanh(W_mat_b)
    # 隐藏层
    with tf.name_scope('Hidden_layer'):
        with tf.name_scope('L1_drop'):
            L1_drop = tf.nn.dropout(L1, keep_prob)
        with tf.name_scope('Weights_output'):
            W_output = tf.Variable(tf.truncated_normal([150, 2], stddev=0.1), dtype=tf.float32, name='weight2')
        with tf.name_scope('biases_output'):
            b_output = tf.Variable(tf.zeros([2])+0.1, dtype=tf.float32, name='biases2')
        with tf.name_scope('W_mat_b_1'):
            W_mat_b_1 = tf.matmul(L1_drop, W_output) + b_output
    # 输出层
    with tf.name_scope('output_layer'):
        with tf.name_scope('prediction'):
            prediction = tf.nn.sigmoid(W_mat_b_1)

    # 损失函数
    with tf.name_scope('loss_fun'):
        with tf.name_scope('W'):
            W = tf.constant([[2, 2], [0.5, 2]], dtype=tf.float32)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(y-prediction)) + layers.l2_regularizer(1e-4)(W)
            tf.summary.scalar('loss', loss)
    # 梯度递减学习率
    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(2.5, global_step=1000, decay_rate=0.85, decay_steps=900)
        tf.summary.scalar('learning_rate', learning_rate)
    # 优化器
    with tf.name_scope('train_step'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    tf.get_collection('prediction', prediction)
    tf.get_collection('cost', loss)
    # 对比真实值和预测值
    with tf.name_scope('accuracy-all'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        # 求准确率
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('logs_s/', sess.graph)
        for eco in range(1001):
            rand_index = np.random.choice(len(train_data), bath_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: train_data[rand_index],
                                                                   y:train_label[rand_index], keep_prob:0.8})
            writer.add_summary(summary, eco)
            acc_test = sess.run(accuracy, feed_dict={x: ver_data, y: ver_label, keep_prob:1.0})
            loss_te = sess.run(loss, feed_dict={x: ver_data, y: ver_label, keep_prob:1.0})
            if eco % 100 == 0:
                print('Iter-->', eco, 'Accuracy-->', acc_test, 'Error-->', loss_te)
        save_path = saver.save(sess, 'save_model/model.ckpt', global_step=1000, write_meta_graph=False)
        print('save_path:', save_path)
        
        
def main():
    file_path = 'new_data.csv'
    file_data = load_data(file_path)
    plot_feature(file_data)
    data = feature_processing(file_data)
    plot_feature_score(data)
    # X, y = split_data(data)
    # X_std = stand_data(X)
    # create_model(X_std, y)


if __name__ == '__main__':
    main()
