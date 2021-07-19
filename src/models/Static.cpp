
// Copyright 2017-2021 Lawrence Livermore National Security, LLC and other
// Apollo Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <string>
#include <cstring>

#include "apollo/models/Static.h"

int
Static::getIndex(std::vector<float> &features)
{
    return policy_choice;
}
