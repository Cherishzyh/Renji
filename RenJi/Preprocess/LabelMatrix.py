import os
import random
from random import shuffle
import pandas as pd


def GenerateCSV():
    preprocess_path = r'D:\Data\renji0723\ProcessedData\Nii'
    label_class = ['0', '1', '2', '3']
    case_dict = {}
    for label in label_class:
        label_folder = os.path.join(preprocess_path, label)
        for case in os.listdir(label_folder):
            case_dict[case] = int(label)
    df = pd.DataFrame({'CaseName': case_dict.keys(), 'Label': case_dict.values()})
    df.to_csv(os.path.join(preprocess_path, 'label.csv'), index=False)
# GenerateCSV()


def DeleteRepeatCase(label_csv):
    df = pd.read_csv(label_csv)
    new_df = df.drop_duplicates()
    new_df.to_csv(r'D:\Data\renji0722\ProcessData\label_new.csv', index=False)
# DeleteRepeatCase()


def CompareCSV():
    a = pd.read_csv(r'C:\Users\82375\Desktop\statistics.csv', encoding='gbk')
    b = pd.read_csv(r'D:\Data\renji0722\ProcessedData\label_new.csv', encoding='gbk')

    a_list = a.values.tolist()
    b_list = b.values.tolist()

    new_a_list = [f[0] for f in a_list]
    new_b_list = [f[0] for f in b_list]

    both = list(set(new_a_list).intersection(set(new_b_list)))
    print(both)
# CompareCSV()


def ConcatCSV():
    case_list = os.listdir(r'D:\Data\renji\Npy')
    case_list = [case[: case.index('.npy')] for case in case_list]
    path1 = r'D:\Data\renji\label_0721.csv'
    path2 = r'D:\Data\renji\label_0722.csv'
    path3 = r'D:\Data\renji\label_0723.csv'
    df1 = pd.read_csv(path1, encoding='gbk')
    df2 = pd.read_csv(path2, encoding='gbk')
    df3 = pd.read_csv(path3, encoding='gbk')
    df = df1
    df = df.append(df2, ignore_index=True)
    df = df.append(df3, ignore_index=True)
    df.drop_duplicates(inplace=True)
    case_name = []
    label = []
    for index in df.index:
        case = df.loc[index, 'CaseName']
        case = '{} {}'.format(case.split(' ')[0], case.split(' ')[1])
        if case in case_list:
            if case not in case_name:
                case_name.append(case)
                label.append(int(df.loc[index, 'Label']))
            else:
                if not int(df.loc[index, 'Label']) == label[case_name.index(case)]:
                    # continue
                    print(case)
    new_df = pd.DataFrame({'CaseName': case_name, 'Label': label})
    new_df.to_csv(r'D:\Data\renji\label_0805.csv', index=False)
    # print()
# ConcatCSV()


def SplitDataset():
    # 同一个患者不同时间的数据应该放在同一个数据集中，不然会导致数据泄露，包括在cv的过程中

    csv_path = r'D:\Data\renji\label.csv'
    df = pd.read_csv(csv_path, index_col='CaseName')
    case_list = df.index.tolist()
    case_list_name = [case.split(' ')[-1] for case in case_list]  # 不重复的数据
    repeat_case_name = [case for case in case_list_name if case_list_name.count(case) > 1]

    train = []
    test = []
    shuffle(case_list)
    train_num = int(len(case_list) * 0.8)
    test_num = int(len(case_list) - train_num)
    for index, case in enumerate(case_list):
        case_name = case.split(' ')[-1]
        if case_name in repeat_case_name:
            if case_name in train:
                train.append(case)
            elif case_name in test:
                test.append(case)
            else:
                if len(train) < train_num:
                    train.append(case)
                else:
                    test.append(case)
        else:
            if len(train) < train_num:
                train.append(case)
            else:
                test.append(case)
    train_name = [case.split(' ')[-1] for case in train]
    test_name = [case.split(' ')[-1] for case in test]
    if len([case for case in train_name if case in test_name]):
        print('Split successful!')
        print('train case: {}; test case: {}'.format(len(train), len(test)))
        train_df = pd.DataFrame({'CaseName': train}).T
        train_df.to_csv(r'D:\Data\renji\train_name.csv')
        test_df = pd.DataFrame({'CaseName': test}).T
        test_df.to_csv(r'D:\Data\renji\test_name.csv')
# SplitDataset()


def SplitCV(cv_folder):
    # 同一个患者不同时间的数据应该放在同一个cv中，不然会导致数据泄露
    csv_path = r'D:\Data\renji\train_name.csv'
    df = pd.read_csv(csv_path)
    case_list = df.values.tolist()[0]
    shuffle(case_list)
    case_list_name = [case.split(' ')[-1] for case in case_list]
    repeat_case_name = list(set([case for case in case_list_name if case_list_name.count(case) > 1])) # 重复的数据
    no_repeat_case_list = [case for case in case_list if case.split(' ')[-1] not in repeat_case_name]
    repeat_case_list = [case for case in case_list if case.split(' ')[-1] in repeat_case_name]

    cv_case_num = len(case_list) // cv_folder
    cv_list = [[] for index in range(cv_folder)]
    cv_name_list = [[] for index in range(cv_folder)]

    cv = 0
    for case in no_repeat_case_list:
        cv_list[cv].append(case)
        cv_name_list[cv].append(case.split(' ')[-1])
        if len(cv_list[cv]) >= len(no_repeat_case_list) // cv_folder:
            if cv == 4:
                continue
            else:
                cv += 1

    cv = 0
    for case in repeat_case_list:
        add = False
        case_name = case.split(' ')
        for index in range(cv_folder):
            if case_name in cv_name_list[index]:
                cv_list[cv].append(case)
                cv_name_list[cv].append(case_name)
                add = True
        if not add:
            cv_list[cv].append(case)
            cv_name_list[cv].append(case_name)
        if len(cv_list[cv]) >= cv_case_num:
            if cv == 4:
                continue
            else:
                cv += 1
    print()

    for index in range(cv_folder):
        for case in cv_name_list[index]:
            for i in range(index+1, cv_folder):
                if case in cv_name_list[i]:
                    raise Exception

    for index in range(cv_folder):
        df = pd.DataFrame({'CaseName': cv_list[index]})
        df.to_csv(r'D:\Data\renji\train-cv{}.csv'.format(index+1), index=False)
# SplitCV(5)


def Statistics(data_path):
    label_csv = r'D:\Data\renji\label.csv'
    label_df = pd.read_csv(label_csv, index_col='CaseName')

    dataset_df = pd.read_csv(data_path)
    data_list = dataset_df.values.tolist()[0]
    if len(data_list) == 1:
        data_list = dataset_df.loc[:, 'CaseName'].tolist()
    label1 = label2 = label3 = label0 = 0
    for case in data_list:
        if label_df.loc[case]['Label'] == 0:
            label0 += 1
        elif label_df.loc[case]['Label'] == 1:
            label1 += 1
        elif label_df.loc[case]['Label'] == 2:
            label2 += 1
        elif label_df.loc[case]['Label'] == 3:
            label3 += 1
        else:
            print(case)
    print('label 0: {}\tlabel 1: {}\tlabel 2: {}\tlabel 3: {}'.format(label0, label1, label2, label3))

def TestStat():
    train_path = r'D:\Data\renji\train_name.csv'
    cv1_train_path = r'D:\Data\renji\train-cv1.csv'
    cv2_train_path = r'D:\Data\renji\train-cv2.csv'
    cv3_train_path = r'D:\Data\renji\train-cv3.csv'
    cv4_train_path = r'D:\Data\renji\train-cv4.csv'
    cv5_train_path = r'D:\Data\renji\train-cv5.csv'
    test_path = r'D:\Data\renji\test_name.csv'

    data_path = [train_path, test_path, cv1_train_path, cv2_train_path, cv3_train_path, cv4_train_path, cv5_train_path]
    for path in data_path:
        Statistics(path)
TestStat()