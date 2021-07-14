// Copyright 2017-2021 Lawrence Livermore National Security, LLC and other
// Apollo Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

#ifndef APOLLO_MODELS_DECISIONTREE_H
#define APOLLO_MODELS_DECISIONTREE_H

#include <string>
#include <vector>

#include "apollo/PolicyModel.h"
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

class DecisionTree : public PolicyModel {

    public:
        DecisionTree(int num_policies, std::vector< std::vector<float> > &features, std::vector<int> &responses);
        DecisionTree(int num_policies, std::string path);

        ~DecisionTree();

        int  getIndex(void);
        int  getIndex(std::vector<float> &features);
        void store(const std::string &filename);
        void load(const std::string &filename);

        std::string generateSource(const std::string &language, const std::string &regionName);

    private:
        Ptr<RTrees> dtree;

        // Internals for the generateSource() function's recursive dive:
        void generateCPPSourceHeader(std::stringstream& code, const std::string &regionName);
        void generateCPPSourceFooter(std::stringstream& code);
        void generateCPPSourceForTree(std::stringstream& code, int rootidx) const;
        void generateCPPSourceForNode(std::stringstream& code, int nodeidx, int depth) const;
        void generateCPPSourceForSplit(std::stringstream& code, int splitidx) const;

        //Ptr<DTrees> dtree;
        //Ptr<SVM> dtree;
        //Ptr<NormalBayesClassifier> dtree;
        //Ptr<KNearest> dtree;
        //Ptr<Boost> dtree;
        //Ptr<ANN_MLP> dtree;
        //Ptr<LogisticRegression> dtree;
}; //end: DecisionTree (class)


#endif
