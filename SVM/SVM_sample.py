from SVM import dataLoading, data_preprocessing, missing_data_processing

from math import exp
import math
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# 파일 읽어오기
from sklearn.model_selection import train_test_split

train, test = dataLoading()
x = data_preprocessing(train, test)

print(test[pd.isna(test["Fare"])])

# test[]

feature_names = ["Pclass", "Sex", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]

X_train = train[feature_names]
Y_train = train["Survived"]

X_test = test[feature_names]

# ########################################################################################################################
# # 러닝 모델 생성
# # DT = 기본 예문
#
# from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
# from sklearn.svm import SVC
# from sklearn import metrics, preprocessing
# from scipy.stats import itemfreq
#
# # X = preprocessing.scale(X)
# X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(X_train, Y_train, test_size=0.3, random_state=3)
#
# c_list = list()
# g_list = list()
#
# c_x_list = list()
# g_x_list = list()
#
# for x in np.arange(-7, 7, 0.5):
#     # c_list.append(10**x)
#     # g_list.append(10**x)
#     c_x_list.append(x)
#     g_x_list.append(x)
#     c_list.append(exp(x))
#     g_list.append(exp(x))
#
# C_grid = c_list #[0.001, 0.01, 0.1, 1, 10]
# gamma_grid = g_list #[0.001, 0.01, 0.1, 1]
# parameters = {'C': C_grid, 'gamma' : gamma_grid}
#
# model = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10)
# model.fit(X_train_cv, Y_train_cv)
# # print(X_train_cv, type(X_train_cv), Y_train_cv, type(Y_train_cv))
#
# best_C = model.best_params_['C']
# best_gamma = model.best_params_['gamma']
#
# print("SVM best C : " + str(best_C))
# print("SVM best gamma : " + str(best_gamma))
# #######################################################################################################################
# # SVM 적용 ############################################################################################################
# # 여기서 부터 볼 것!!!
# # 시각화 및 출력데이터 정리
# model_SVM = SVC(C=best_C,gamma=best_gamma)
# model_SVM.fit(X_train_cv, Y_train_cv)
#
# prediction = model_SVM.predict(X_test)
#
# # # 테스트 (예측)
#
# submission = pd.read_csv("data/gender_submission.csv", index_col="PassengerId")
# submission["Survived"] = prediction
# # print(submission.shape, type(submission))
#
# result_file = "result/result_SVM.csv"
# submission.to_csv(result_file, mode='w')
#
# ########################################################################################################################
# # 여기서 확인할 수 있는 최대 정확도는 0.77 아직 pca 미적용 #############################################################
#
# for c in c_list:
#     for g in g_list:
#         model_SVM = SVC(C=c, gamma=best_gamma)
#         model_SVM.fit(X_train_cv, Y_train_cv)
#         prediction = model_SVM.predict(X_test_cv)
#
#         accuracy = metrics.accuracy_score(Y_test_cv, prediction)
#         print('Accuracy    = ' + str(np.round(accuracy, 2)))
# ########################################################################################################################



# confusion matrix 계산
# 테스트 데이터의 정답이 존재하지 않으므로 의미없다

# def confusion_matrix(pred, gt):
#     cont = np.zeros((2,2))
#     for i in [0, 1]:
#         for j in [0, 1]:
#             cont[i, j] = np.sum((pred == i) & (gt == j))
#     return cont

# survived == prediction

# print(cfm)
# import matplotlib.pyplot as plt