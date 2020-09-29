
// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// This file is part of Apollo.
// OCEC-17-092
// All rights reserved.
//
// Apollo is currently developed by Chad Wood, wood67@llnl.gov, with the help
// of many collaborators.
//
// Apollo was originally created by David Beckingsale, david@llnl.gov
//
// For details, see https://github.com/LLNL/apollo.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#ifndef APOLLO_CONFIG_H
#define APOLLO_CONFIG_H

namespace Apollo
{

class Config {
    public:
        ~Config();

        void loadSettings(void);
        bool sanityCheck(bool abort_on_fail);

        int APOLLO_COLLECTIVE_TRAINING;
        int APOLLO_LOCAL_TRAINING;
        int APOLLO_SINGLE_MODEL;
        int APOLLO_REGION_MODEL;
        int APOLLO_TRACE_MEASURES;
        int APOLLO_NUM_POLICIES;
        int APOLLO_TRACE_POLICY;
        int APOLLO_RETRAIN_ENABLE;
        float APOLLO_RETRAIN_TIME_THRESHOLD;
        float APOLLO_RETRAIN_REGION_THRESHOLD;
        int APOLLO_STORE_MODELS;
        int APOLLO_TRACE_RETRAIN;
        int APOLLO_TRACE_ALLGATHER;
        int APOLLO_TRACE_BEST_POLICIES;
        std::string APOLLO_INIT_MODEL;
        std::string APOLLO_LOAD_MODEL;

    private:
        Config();

    friend class Apollo::Exec;
}; //end: Config (class)
}; //end: Apollo (namespace)

#endif
