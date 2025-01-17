
// Copyright 2017-2021 Lawrence Livermore National Security, LLC and other
// Apollo Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <sys/stat.h>

#include "apollo/models/DecisionTree.h"
#include <opencv2/core/types.hpp>

#define modelName "decisiontree"
#define modelFile __FILE__


using namespace std;

static inline bool fileExists(std::string path) {
    struct stat stbuf;
    return (stat(path.c_str(), &stbuf) == 0);
}

DecisionTree::DecisionTree(int num_policies, std::string path)
    : PolicyModel(num_policies, "DecisionTree", false)
{
    if (not fileExists(path)) {
        std::cerr << "== APOLLO: Cannot access the DecisionTree model requested:\n" \
                  << "== APOLLO:     " << path << "\n" \
                  << "== APOLLO: Exiting.\n";
        exit(EXIT_FAILURE);
    } else {
        // The file at least exists... attempt to load a model from it!
        std::cout << "== APOLLO: Loading the requested DecisionTree:\n" \
                  << "== APOLLO:     " << path << "\n";
        dtree = RTrees::load(path.c_str());
    }
    return;
}

DecisionTree::DecisionTree(int num_policies, std::vector< std::vector<float> > &features, std::vector<int> &responses)
    : PolicyModel(num_policies, "DecisionTree", false)
{

    //std::chrono::steady_clock::time_point t1, t2;
    //t1 = std::chrono::steady_clock::now();
    //int training_set_depth = features.size();

    //dtree = NormalBayesClassifier::create();
    //
    //dtree = KNearest::create();
    //
    //dtree = Boost::create();
    //
    //dtree = ANN_MLP::create();
    //
    //dtree = SVM::create();
    //dtree = LogisticRegression::create();
    //dtree->setLearningRate(0.001);
    //dtree->setIterations(10);
    //dtree->setRegularization(LogisticRegression::REG_L2);
    //dtree->setTrainMethod(LogisticRegression::BATCH);
    //dtree->setMiniBatchSize(1);

    //dtree = DTrees::create();
    dtree = RTrees::create();
    dtree->setTermCriteria( TermCriteria(  TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 0.01 ) );
    //dtree->setTermCriteria( TermCriteria(  TermCriteria::MAX_ITER, 10, 0 ) );

    dtree->setMaxDepth(2);

    dtree->setMinSampleCount(1);
    dtree->setRegressionAccuracy(0);
    dtree->setUseSurrogates(false);
    dtree->setMaxCategories(policy_count);
    dtree->setCVFolds(0);
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(false);
    dtree->setPriors(Mat());

    Mat fmat;
    for(auto &i : features) {
        Mat tmp(1, i.size(), CV_32F, &i[0]);
        fmat.push_back(tmp);
    }

    Mat rmat;
    //rmat = Mat::zeros( fmat.rows, num_policies, CV_32F );
    //for( int i = 0; i < responses.size(); i++ ) {
    //    int j = responses[i];
    //    rmat.at<float>(i, j) = 1.f;
    //}
    Mat(fmat.rows, 1, CV_32S, &responses[0]).copyTo(rmat);
    //Mat(fmat.rows, 1, CV_32F, &responses[0]).copyTo(rmat);

    //std::cout << "fmat: " << fmat << std::endl;
    //std::cout << "features.size: " << features.size() << std::endl;
    //std::cout << "rmat: " << rmat << std::endl;

    // ANN_MLP
    //dtree->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);
    //Mat layers(3, 1, CV_16U);
    //layers.row(0) = Scalar(fmat.cols);
    //layers.row(1) = Scalar(4);
    //layers.row(2) = Scalar(rmat.cols);
    //dtree->setLayerSizes( layers );
    //dtree->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);

    dtree->train(fmat, ROW_SAMPLE, rmat);
    //Ptr<TrainData> data = TrainData::create(fmat, ROW_SAMPLE, rmat);
    //dtree->train(data);
    //for(int i = 0; i<1000; i++)
    //    dtree->train(data, ANN_MLP::TrainFlags::UPDATE_WEIGHTS);

    //if(!dtree->isTrained()) {
    //    std::cout << "MODEL IS NOT TRAINED!" << std::endl;
    //    abort();
    //}


    //t2 = std::chrono::steady_clock::now();
    //double duration = std::chrono::duration<double>(t2 - t1).count();
    //std::cout << "train," << training_set_depth << "," << name<< "," << std::fixed << std::setprecision(12) << duration << "\n";

    return;
}

DecisionTree::~DecisionTree()
{
    return;
}

int
DecisionTree::getIndex(std::vector<float> &features)
{
    //std::chrono::steady_clock::time_point t1, t2;
    //t1 = std::chrono::steady_clock::now();
    //int choice = dtree->predict( features );
    //t2 = std::chrono::steady_clock::now();
    //double duration = std::chrono::duration<double>(t2 - t1).count();
    //std::cout << "predict," << features.size() << "," << choice << "," << std::fixed << std::setprecision(12) << duration << "\n"; //ggout

    //return choice;
    return dtree->predict( features );

}

void DecisionTree::store(const std::string &filename)
{
    dtree->save( filename );
}
