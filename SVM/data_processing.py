import os
import pandas as pd



def dataLoading():
    if os.path.exists("../data"):
        train = pd.read_csv("../data/train.csv")
        test = pd.read_csv("../data/test.csv")
        return train, test

def data_preprocessing(train=None, test=None):
    if train is not None and test is not None:
        # 성별 변수 수치화
        train.loc[train["Sex"] == "male", "Sex"] = 0
        train.loc[train["Sex"] == "female", "Sex"] = 1

        test.loc[test["Sex"] == "male", "Sex"] = 0
        test.loc[test["Sex"] == "female", "Sex"] = 1

        # 리스트로 목록 갯수와 맞춰서 넣어야 하지 않나???
        train["Embarked_C"] = train["Embarked"] == "C"
        train["Embarked_S"] = train["Embarked"] == "S"
        train["Embarked_Q"] = train["Embarked"] == "Q"

        test["Embarked_C"] = test["Embarked"] == "C"
        test["Embarked_S"] = test["Embarked"] == "S"
        test["Embarked_Q"] = test["Embarked"] == "Q"
    else:
        x = dataLoading()
        return x

def missing_data_processing(df, method='mean'):

    if method == 'mean':
        mean_fare = df["Fare"].mean()
    # 데이터 전처리 ########################################################################################################
    ########################################################################################################################
    # fare (요금) 컬럼의 결측치를 평균값으로 채움

    # print("Fare(Mean) = ${0:.3f}".format(mean_fare))
    test.loc[pd.isnull(test["Fare"]), "Fare"] = mean_fare

    ########################################################################################################################