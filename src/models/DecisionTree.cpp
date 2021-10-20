
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
#include <opencv2/ml.hpp>

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


void
DecisionTree::generateCPPSourceRandomForest(std::stringstream& code, int numPolicies)
{
    //NOTE[cdw]: 'dtree' is a private class member, see .h file.
    const std::vector<int>& roots = dtree->getRoots();
    const std::vector<cv::ml::DTrees::Node>& nodes = dtree->getNodes();
    const std::vector<cv::ml::DTrees::Split>& splits = dtree->getSplits();

    code << "static const std::vector<int> rootNodes { ";
    for (auto root : roots) {
        code << root << ", ";
    }
    code << " };\n\n";

    code << "static const std::vector<int> nodeSplitIndex { ";
    for (auto node : nodes) {
        code << node.split << ", ";
    }
    code << " };\n\n";

    code << "static const std::vector<int> nodeLeft { ";
    for (auto node : nodes) {
        code << node.left << ", ";
    }
    code << " };\n\n";

    code << "static const std::vector<int> nodeRight { ";
    for (auto node : nodes) {
        code << node.right << ", ";
    }
    code << " };\n\n";

    code << "static const std::vector<int> nodeSuggestedPolicy { ";
    for (auto node : nodes) {
        code << node.classIdx << ", ";
    }
    code << " };\n\n";

    code << "static const std::vector<int> splitOnVariableIdx { ";
    for (auto split : splits) {
        code << split.varIdx << ", ";
    }
    code << " };\n\n";

    code << "static const std::vector<float> splitAtThreshold { ";
    for (auto split : splits) {
        code << split.c << ", ";
    }
    code << " };\n\n";

    code << "static const std::vector<bool> splitDirectionReversed { ";
    for (auto split : splits) {
        code << (int) split.inversed << ", ";
    }
    code << " };\n\n";

    code << "static std::vector<int> votes(" << numPolicies << ");\n";
    code << \
R"(
    // Clear out votes from previous prediction:
    std::fill(votes.begin(), votes.end(), 0);

    // These are filled-in/used in the loops below as we walk the
    // trees of the forest and a
    int node_split_idx;
    int node_child_right;
    int node_child_left;
    int node_recommended_pol;
    int   split_on_var_idx;
    float split_at_threshold;
    bool  split_is_gt;

    for (int ridx: rootNodes) {
        int nidx = ridx;
        int prev = nidx;

        for (;;)
        {
            //NOTE[cdw]: Terminal leaves in the tree encoding have a
            //           sentinel value where node_split_idx < 0,
            //           which is why we "memoize" 'prev' for use when
            //           extracting policy recommendations, after
            //           arriving at the terminating sentinal node.

            prev = nidx;

            node_split_idx   = nodeSplitIndex[nidx];
            node_child_right = nodeRight[nidx];
            node_child_left  = nodeLeft[nidx];

            if( node_split_idx < 0 ) {
                // This is a leaf node, it refers to no deeper branches.
                break;
            }

            split_on_var_idx   = splitOnVariableIdx[node_split_idx];
            split_at_threshold = splitAtThreshold[node_split_idx];
            split_is_gt        = splitDirectionReversed[node_split_idx];

            float val = immediateFeatureValues[split_on_var_idx];

            //NOTE[cdw]: In OpenCV...
            //               if( vtype[va] == VAR_ORDERED ) ...
            //           the below line is the only logic, it does not
            //           check 'split_is_gt' ... that only comes up when
            //               float val = psample[ci*sstep];
            //               if (val == MISSED_VAL ) {
            //                  nidx = (split_is_gt ? node_child_right : node_child_left);
            //                  continue;
            //               }
            //               val = missingSubstPtr[vi];
            //               ...
            //
            //    TODO[cdw]:
            //           SO, I think we're missing a step here, where
            //           somehow we get to a point in the tree and
            //           nothing is true, and we've not yet made a
            //           recommendation, and we need to bump right instead
            //           of left, or something like that. Not sure.
            //
            //
            nidx = (val <= split_at_threshold ? node_child_left : node_child_right);
        }

        node_suggested_pol = nodeSuggestPolicy[prev];
        votes[node_suggested_pol]++;

        } //end: parsing this tree
    } //end: parsing all trees in forest

    // Analyze the votes and return the winner!
    int best_idx = node_suggested_pol;
    if( roots.size() > 1 )
    {
        best_idx = 0;
        for(int i = 1; i < )" << numPolicies << R"(; i++ ) {
            if( votes[best_idx] < votes[i] )
            best_idx = i;
        }
    }



)";



    return;
}





void
DecisionTree::generateCPPSourceHeader(std::stringstream& code, const std::string& regionName)
{
    return;
}


void
DecisionTree::generateCPPSourceFooter(std::stringstream& code)
{
    return;
}



std::string
DecisionTree::generateSource(const std::string &language, const std::string &regionName)
{
    // NOTE[cdw]: For now, we only support C++ code generation.
    if (not ((language == "c++") || \
             (language == "C++") || \
             (language == "CPP") || \
             (language == "Cpp") || \
             (language == "cpp")))
    {
        std::cerr << "== APOLLO: DecisionTree::generateSource(" \
                  << language << ") is not supported.\n" \
                  << "== APOLLO: Returning empty string.\n";
        return "";
    }

    std::stringstream code;
    const std::vector<int>&  roots = dtree->getRoots();
    const std::vector<cv::ml::DTrees::Node>& nodes = dtree->getNodes();


    // OK, here we go...
    std::cout << "roots.size() == " << roots.size() << "\n";
    for (auto rootIdx : roots) {
        std::cout << "roots[" << rootIdx << "] == " \
                  << roots[rootIdx] << "\n";
    }

    generateCPPSourceHeader(code, regionName);
    generateCPPSourceRandomForest(code, policy_count);
    generateCPPSourceFooter(code);

    std::cout << "---------- GENERATED CODE ----------\n" \
              << code.str() << "\n" \
              << "------------------------------END---\n";

    return code.str();

}
