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

    train0set = []
    train1set = []

    # extracting datapoints into 2d arrays as [mean,variance]
    for i in range(len(train0)):  # for both tests since train0 and train1 have the same lenght
        train0set.append([numpy.mean(train0[i]), numpy.std(train0[i])])
        train1set.append([numpy.mean(train1[i]), numpy.std(train1[i])])

    meantrain0 = numpy.mean(train0set, axis=0)
    vartrain0 = numpy.var(train0set, axis=0)
    meantrain1 = numpy.mean(train1set, axis=0)
    vartrain1 = numpy.var(train1set, axis=0)

    mf1train0 = meantrain0[0]
    vf1train0 = vartrain0[0]
    mf2train0 = meantrain0[1]
    vf2train0 = vartrain0[1]
    mf1train1 = meantrain0[0]
    vf1train1 = vartrain1[0]
    mf2train1 = meantrain1[1]
    vf2train1 = vartrain1[1]

    samplespredicted0 = 0
    feature1test0 = []
    feature2test0 = []
    for t in range(len(test0)):
        feature1test0.append(numpy.mean(test0[t]))
        feature2test0.append(numpy.std(test0[t]))

    feature1test1 = []
    feature2test1 = []
    for r in range(len(test1)):
        feature1test1.append(numpy.mean(test1[r]))
        feature2test1.append(numpy.std(test1[r]))

    # train0d = extract_features(train0)
    # train0d = numpy.reshape(train0, (-1, 28))
    # train1d = extract_features(train1)
    # train1d = numpy.reshape(train1, (-1, 28))

    '''feature1train0 = numpy.mean(train0, axis=0)
    feature2train0 = numpy.std(train0d, axis=1)
    feature1train1 = numpy.mean(train1, axis=0)
    feature2train1 = numpy.std(train1d, axis=1)

    mf1train0 = sum(feature1train0)/len(feature1train0)
    mf2train0 = sum(feature2train0)/len(feature2train0)
    mf1train1 = sum(feature1train1)/len(feature1train1)
    mf2train1 = sum(feature2train1)/len(feature2train1)
    vf1train0 = variance(feature1train0)
    vf2train0 = variance(feature2train0)
    vf1train1 = variance(feature1train1)
    vf2train1 = variance(feature2train1)
    vf1train0 = numpy.var(feature1train0)
    vf1train1 = numpy.var(feature1train1)
    vf2train0 = numpy.var(feature2train0)
    vf2train1 = numpy.var(feature2train1)
    mf1train0 = numpy.mean(feature1train0)
    mf2train0 = numpy.mean(feature2train0)
    mf1train1 = numpy.mean(feature1train1)
    mf2train1 = numpy.mean(feature2train1)'''

    '''for f in range(len(feature1test0)):
        probf1train0 = NB(feature1test0[f],mf1train0,numpy.std(feature1train0))
        probf2train0 = NB(feature2test0[f],mf2train0,numpy.std(feature2train0))
        expectedprobtrain0 = probf1train0*probf2train0*0.5

        probf1train1 = NB(feature1test0[f],mf1train1,numpy.std(feature1train1))
        probf2train1 = NB(feature2test0[f],mf2train1,numpy.std(feature2train1))
        expectedprobtrain1 = probf1train1*probf2train1*0.5

        #e

        if expectedprobtrain0>expectedprobtrain1:
            samplespredicted0 += 1'''

    Accuracy_for_digit0testset = 0
    Accuracy_for_digit1testset = 0

    # print (Accuracy_for_digit0testset)

    # implement NB calssifiers parameters from task 2; use classifiers to predict unknown labels
    # predict labels for test set?

    print(['1670', mf1train0, vf1train0, mf2train0, vf2train0, mf1train1, vf1train1, mf2train1, vf2train1,
           Accuracy_for_digit0testset, Accuracy_for_digit1testset])
    # ['ASUId', Mean_of_feature1_for_digit0, Variance_of_feature1_for_digit0, Mean_of_feature2_for_digit0, Variance_of_feature2_for_digit0 , Mean_of_feature1_for_digit1, Variance_of_feature1_for_digit1, Mean_of_feature2_for_digit1, Variance_of_feature2_for_digit1, Accuracy_for_digit0testset, Accuracy_for_digit1testset]

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


# NB classifier formula
def NB(test, avg, omega):
    exponent = -0.5 * (((test - avg) / omega) ** 2)
    e = math.e ** (exponent)
    denominator = omega * math.sqrt(2 * math.pi)
    return e / denominator


if __name__ == '__main__':
    main()