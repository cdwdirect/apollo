// Copyright 2017-2021 Lawrence Livermore National Security, LLC and other
// Apollo Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef APOLLO_TIMING_MODEL_H
#define APOLLO_TIMING_MODEL_H

#include <string>
#include <vector>

// Abstract
class TimingModel {
    protected:
        std::string
        generateDefaultSource(const std::string &language) {
            // NOTE[cdw]: Eventually we may wish to add any boilerplate
            //            macros here that get emplaced during source
            //            gen of different kinds of models. These are
            //            places where a consumer of this stringified
            //            model can insert their own interface code
            //            using standard search/replace.
            return "";
        }
    public:
        TimingModel(std::string name) : name(name) {};
        virtual ~TimingModel() {}

        virtual double
            getTimePrediction(std::vector<float> &features) = 0;

        virtual void
            store(const std::string &filename) = 0;

        virtual std::string
            generateSource(const std::string &language) {
                return generateDefaultSource(language);
            }

        std::string name = "";

}; //end: TimingModel (abstract class)


#endif
