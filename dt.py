
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
from pprint import pprint
from sys import argv
import time

start = time.time()
train_path = sys.argv[1]
test_path = sys.argv[2]
result_path = sys.argv[3]
train_file = open(train_path, "r")
test_file = open(test_path, "r")

train_data = pd.read_csv(train_file, sep='\t')
test_data = pd.read_csv(test_file, sep='\t')

train_attributes = list(train_data.columns)  # 전체 column얻기
target_column = train_attributes.pop()  # class를 결정하는 target column 따로 분리
class_list = np.unique(train_data[target_column])  # class를 나누기
test_attributes = []  # infoGain으로 정해지는 atrributes을 리스트로 만들기

# 해당 element가 데이터에서 차지하는 확률을 구하는 함수
# elements인자는 리스트 형태로 들어오고, 리스트는 서로 다른 속성들의 개수를 포함(<=30[yes : 3, no : 2])
# 각 속성들의 개수(elements[i])를 전체 개수(np.sum(elements))로 나누면 해당 element의 확률


def get_prob(elements):
    p = []
    for i in range(len(elements)):
        p.append(elements[i]/np.sum(elements))
    return p

# entropy, 즉 heterogeneous의 정도를 찾아 리턴하는 함수


def get_entropy(an_attribute):
    # 임이의 attibute을 넘겨 받아, 이 attribute이 포함하는 데이터를 중
    # unique한 값들의 리스트와 그 개수를 리턴
    unique_elements, counts = np.unique(an_attribute, return_counts=True)
    # unique한 값들의 개수를 포함한 리스트를 이용해 확률 계산
    p = get_prob(counts)
    # 각각의 값들에 대한 확률은 리스트로 전달되어
    # 확률 리스트를 인덱싱하면서 entropy 계산
    entropy = 0
    for i in range(len(unique_elements)):
        entropy += -(p[i] * np.log2(p[i]))
    return entropy

# information_gain을 얻는 함수
# 트리의 각 branch 데이터와 그 sub root, 타겟을 받아서
# 전체 엔트로피와 자식노드들의 엔트로피를 구해서 info_gain 리턴


def get_InfoGain(data, sub_root, target_name):
    # 전체 엔트로피 계산
    total_entropy = get_entropy(data[target_name])

    # 각 subTree 의 엔트로피 계산
    # unique한 attribute의 리스트를 추출
    sub_trees, counts = np.unique(data[sub_root], return_counts=True)
    # 각각의 리스트에 대한 확률 계산
    p = get_prob(counts)
    # 각각의 attribute에 대한 확률에 그 attribute의 엔트로피를 곱한 값들의 합이
    # sub_root의 전체 info-gain이 된다.
    # dataframe.where를 이용해 전체 sub tree에서 각 branch에 해당하는 값들만 추출
    # where 조건에 해당하지 않는 나머지 값들(Nan)은 dropna()를 이용해 제거
    # 남은 데이터 중 target_column이 가지고 있는 데이터만 추출하여 entropy함수에 인자로 전달
    sub_entropy = 0
    for i in range(len(sub_trees)):
        sub_entropy += p[i] * get_entropy(
            data.where(data[sub_root] == sub_trees[i]).dropna()[target_name])

    Info_Gain = total_entropy - sub_entropy
    return Info_Gain

# 트리를 만드는 함수
# data : 전체 데이터에서 recursive과정에 subtree로 분리되는 데이터
# originaldata : 변하지 않는 전체 데이터
# target_attribute : 모델을 정하는 기준 값(column), 트리의 leaf에 위치
# train_attributes : target_attribute을 제외한 나머지 column들
# parent_node : 첫 파라미터는 root이므로 parent_node를 None으로 설정


def BuildTree(data, originaldata, train_attributes, target_attribute, parent_node=None):

    # no samples left의 경우 parent 반환
    if len(train_attributes) == 0:
        return parent_node
    # all samples are in the same class
    # 해당 데이터의 target attribute 값을 반환
    elif len(np.unique(data[target_attribute])) <= 1:
        return np.unique(data[target_attribute])[0]

    # no remaining attributes
    # 이 경우 original data에서 최대다수를 이루는 값 반환
    elif len(data) == 0:
        # original_data에서 unique한 값들의 개수리스트를 추출
        unique_counts = np.unique(
            originaldata[target_attribute], return_counts=True)[1]
        # 그 중 가장 큰 값의 인덱스를 추출
        max_index = np.argmax(unique_counts)
        # original-data를 다시 인덱싱하여 최대다수의 값을 리턴
        return np.unique(originaldata[target_attribute])[max_index]

    # 트리 키우기
    else:
        # define the attribute of parent node
        # 부모노드는 전체 데이터에서 최대다수에 포함되는 속성으로 정의(yes or no or others)
        max_index = np.argmax(
            np.unique(originaldata[target_attribute], return_counts=True)[1])
        parent_node = np.unique(originaldata[target_attribute])[max_index]

        # 각각의 속성들을 선택했을 때의 information gain을 얻어 리스트로 저장
        infoGain_list = []
        for attribute in train_attributes:
            infoGain_list.append(get_InfoGain(
                data, attribute, target_attribute))
        # 이중 가장 큰 값(infoGain)의 인덱스를 추출하여, 그 인덱스에 해당하는  attribute을 root로 지정
        best_attr_index = np.argmax(infoGain_list)
        root = train_attributes[best_attr_index]
        test_attributes.append(root)
        # root을 이용해 json구조로 트리 생성
        tree = {root: {}}

        # root을 하나씩 제거하면서 recursive 실행 시 데이터를 분할(divide & conquer)
        for i in train_attributes:
            if i == root:
                train_attributes.remove(i)

        # recursive를 실행하면서 branch 키우기
        for branch in np.unique(data[root]):
            # root가 포함한 값들을 unique을 기준으로 클러스터링
            # df.where 함수를 이용하여 각 branch value에 맞는 값만 sub_data에 저장
            sub_data = data.where(data[root] == branch).dropna()

            # recursive action
            # 추출한 sub_data들로 다시 함수를 반복적으로 호출하면서 트리 키우기
            subtree = BuildTree(sub_data, data, train_attributes,
                                target_attribute, parent_node)
            tree[root][branch] = subtree

        return(tree)

# 만든 tree와 test data를 이용하여 결과를 예측해보기


def Predict(tree, dic):
    if tree in class_list:
        return tree
    for key in dic.keys():
        if key in tree.keys():
            subtree = tree[key]
            break
        elif dic[key] in tree.keys():
            subtree = tree[dic[key]]
            break

    return Predict(subtree, dic)

# test 해보기
# infoGain으로 얻은 column들의 순서를 target_list로 정하고
# 각 row마다 target_list에 대응하는 값들을 column : value로 저장하고
# 다 저장된 row를 한 줄씩 predict 함수로 보내 result를 예측
# 실행이 끝난 후에 result리스트를 test데이터에 하나의 column으로 concatenate 하기

def Test_tree(tree):
    dic = {}
    result = []
    for i in range(len(test_data)):
        for j in test_attributes:
            dic[j] = test_data[j][i]
        result.append(Predict(tree, dic))
    return result


# main 함수
if __name__ == '__main__':
    # tree 만들기
    tree = BuildTree(train_data, train_data, train_attributes, target_column)
    pprint(tree)

    test_data[target_column] = Test_tree(tree)
    test_data.to_csv("result_data.csv", sep='\t')
    #result_file = open(result_path, "w")
    # result_file.write(test_data.to_string())
    train_file.close()
    test_file.close()
    # result_file.close()

    print("Result output completed!!")
    print("execution time :", str(round((time.time() - start), 1)) + 's')
