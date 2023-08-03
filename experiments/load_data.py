import numpy as np
import pickle

path = 'D:/Files/VisualStudioCode/TT2.0/Ubiquitous-Train/experiments'

def load_data():
    """load data

    Parameters
    ----------
    None

    Returns
    -------
    6 nD-list
        genreMat, occupationMat, ratingTensor, count_rating, ageMat, genderMat
    """
    fr = open(path+"/ml_100k/u.occupation", "r")
    occu_id = 0
    occupationDict = {}
    for info in fr.readlines():#读取file中的所有行，放于一个list中，每行作为一个list元素
        occupationDict[info.strip().split('\t')[0]] = occu_id #每个occupation都给一个对应的occu_id
        occu_id += 1
    fr.close()
    #得到一个字典容器{occupation：occu_id}
    #职业类别数量:21

    fr = open(path+"/ml_100k/u.user", "r")
    occupationMat = np.zeros((943, 21))  #用户数:943 & 职业类别数量:21
    for info in fr.readlines():
        info = info.strip().split("|")
        #strip()移除首尾指定字符串,默认为空格
        #split通过指定分隔符对字符串进行切片,默认为空格
        userId = int(info[0]) - 1  #用于tensor,需从0开始
        occuId = occupationDict[info[3]]  #occu_id
        occupationMat[userId][occuId] = 1
    fr.close()
    #得到tensor：user-occupation

    fr = open(path+"/ml_100k/u.user", "r")
    ageMat = np.zeros((943, 4))  #用户数:943 & 年龄类别:4
    for info in fr.readlines():
        info = info.strip().split("|")
        #strip()移除首尾指定字符串,默认为空格
        #split通过指定分隔符对字符串进行切片,默认为空格
        userId = int(info[0]) - 1  #用于tensor,需从0开始
        age = int(info[1])
        if age in range(0,15):
            ageId = 0
        elif age in range(15,36):
            ageId = 1
        elif age in range(36,65):
            ageId = 2
        elif age in range(65,200):
            ageId = 3
        else:
            print("errorUserId:",userId)
            raise ValueError("Incorrect age format")
        ageMat[userId][ageId] = 1
    fr.close()
    #得到tensor：user-age

    fr = open(path+"/ml_100k/u.user", "r")
    genderMat = np.zeros((943, 2))  #用户数:943 & 性别:2
    for info in fr.readlines():
        info = info.strip().split("|")
        #strip()移除首尾指定字符串,默认为空格
        #split通过指定分隔符对字符串进行切片,默认为空格
        userId = int(info[0]) - 1  #用于tensor,需从0开始
        gender = info[2]
        if gender == "M":
            genderId = 0
        elif gender == "F":
            genderId = 1
        else:
            print("errorUserId:",userId)
            raise ValueError("Incorrect gender format")
        genderMat[userId][genderId] = 1
    fr.close()
    #得到tensor:user-gender


    fr = open(path+"/ml_100k/movies.txt", "r")
    genreMat = np.zeros((1682, 19))  #电影数量：1682 && 电影类型：19
    for info in fr.readlines():
        info = info.strip().split('|')
        movieId = int(info[0]) - 1
        for gId in range(19):
            genreMat[movieId][gId] = int(info[5:][gId])
    fr.close()
    #得到tensor:movieid-mtype

    fr = open(path+"/ml_100k/ratings.txt", "r")
    count_rating = 0  #统计rating数据总行数
    ratingTensor = np.zeros((943, 1682, 5))  #用户数量：943 电影数量：1682 评分：1，2，3，4，5
    for info in fr.readlines():
        info = info.strip().split("\t")
        userId = int(info[0]) - 1
        movieId = int(info[1]) - 1
        ratingId = int(info[2]) - 1
        ratingTensor[userId][movieId][ratingId] = 1
        count_rating = count_rating + 1
    fr.close()
    #print('ratingTensor[userId][movieId][ratingId]:',ratingTensor[5][85])
    #得到tensor：userid-movieid-ratingid
    return genreMat, occupationMat, ratingTensor, ageMat, genderMat

def stroe_tensor(tensorList,tensorNameList):
    tNum = len(tensorList)
    tNameNum = len(tensorNameList)
    if tNum != tNameNum:
        raise ValueError("The elements in tensorList and tensorNameList should correspond one by one")
    for i in range(tNum):
        with open(path+"/t_storage/"+tensorNameList[i]+".bin", 'wb+') as fp: 
            pickle.dump(tensorList[i], fp)

def load_tensor(tensorNameList):
    tNameNum = len(tensorNameList)
    tensorList = []
    for i in range(tNameNum):
        with open(path+"/t_storage/"+tensorNameList[i]+".bin", 'rb') as fp:
            t = pickle.load(fp)
            tensorList.append(t)
    return tensorList


if __name__ == "__main__":
    genreMat,occupationMat,ratingTensor,ageMat,genderMat = load_data()
    tensorList = [genreMat,occupationMat,ratingTensor,ageMat,genderMat]
    tensorNameList = ["genreMat","occupationMat","ratingTensor","ageMat","genderMat"]
    print(tensorNameList)
    stroe_tensor(tensorList,tensorNameList)
    tensorList = load_tensor(tensorNameList)
    for t in tensorList:
        print(np.shape(t))
    
