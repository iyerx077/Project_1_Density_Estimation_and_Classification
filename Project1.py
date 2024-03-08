import numpy
import scipy.io
import math
import geneNewData
import matplotlib.pyplot


# Numpyfile= scipy.io.loadmat("matlabfile.mat")

def main():
    myID = '1670'  # change to last 4 digit of your studentID
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train' + myID + '.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train' + myID + '.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset' + '.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset' + '.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')

    train0d = numpy.array(train0)
    # train0d = numpy.reshape(train0, (-1, 28))
    train1d = numpy.array(train1)
    # train1d = numpy.reshape(train1, (-1, 28))

    # print (numpy.ndim(train0d))

    feature1train0 = numpy.mean(train0d, axis=0)
    feature2train0 = numpy.std(train0d, axis=0)
    feature1train1 = numpy.mean(train1d, axis=0)
    feature2train1 = numpy.std(train1d, axis=0)

    '''mf1train0 = sum(feature1train0)/len(feature1train0)
    mf2train0 = sum(feature2train0)/len(feature2train0)
    mf1train1 = sum(feature1train1)/len(feature1train1)
    mf2train1 = sum(feature2train1)/len(feature2train1)
    vf1train0 = variance(feature1train0)
    vf2train0 = variance(feature2train0)
    vf1train1 = variance(feature1train1)
    vf2train1 = variance(feature2train1)'''
    vf1train0 = numpy.var(feature1train0)
    vf1train1 = numpy.var(feature1train1)
    vf2train0 = numpy.var(feature2train0)
    vf2train1 = numpy.var(feature2train1)
    mf1train0 = numpy.mean(feature1train0)
    mf2train0 = numpy.mean(feature2train0)
    mf1train1 = numpy.mean(feature1train1)
    mf2train1 = numpy.mean(feature2train1)
    Accuracy_for_digit0testset = 0
    Accuracy_for_digit1testset = 0

    result = ['1670', mf1train0, vf1train0, mf2train0, vf2train0, mf1train1, vf1train1, mf2train1, vf2train1,
              Accuracy_for_digit0testset, Accuracy_for_digit1testset]
    # ['ASUId', Mean_of_feature1_for_digit0, Variance_of_feature1_for_digit0, Mean_of_feature2_for_digit0, Variance_of_feature2_for_digit0 , Mean_of_feature1_for_digit1, Variance_of_feature1_for_digit1, Mean_of_feature2_for_digit1, Variance_of_feature2_for_digit1, Accuracy_for_digit0testset, Accuracy_for_digit1testset]
    print(result)
    # print (feature2train0[0])

    '''print (mf1train0)
    print (mf2train0)
    print (mf1train1)
    print (mf2train1)
    print (vf1train0)
    print (vf2train0)
    print (vf1train1)
    print (vf2train1)'''

    '''meantrain0 = numpy.mean(train0)
    meantrain1 = numpy.mean(train1)
    stdtrain0 = numpy.std(train0)
    print (meantrain0)
    print (meantrain1)
    print (stdtrain0)'''
    # each of the matrices in the train0 and train1 has 28 elements

    # print([len(train0),len(train1),len(test0),len(test1)])

    # print('Your trainset and testset are generated successfully!')
    pass


# 1/N*sum(average-element)**2
def variance(array):
    average = sum(array) / len(array)
    var = 0
    for i in range(len(array)):
        var += (average - array[i]) ** 2
    return var / len(array)


if __name__ == '__main__':
    main()