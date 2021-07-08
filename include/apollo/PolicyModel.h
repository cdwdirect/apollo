#ifndef APOLLO_POLICY_MODEL_H
#define APOLLO_POLICY_MODEL_H

#include <string>
#include <vector>

// Abstract
class PolicyModel {
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
        PolicyModel(int num_policies, std::string name, bool training) :
            policy_count(num_policies),
            name(name),
            training(training)
        {};
        virtual ~PolicyModel() {}
        //
        virtual int
            getIndex(std::vector<float> &features) = 0;

        virtual void
            store(const std::string &filename) = 0;

        virtual std::string
            generateSource(const std::string &language) {
                return generateDefaultSource(language);
            }

        int              policy_count;
        std::string      name              = "";
        bool             training          = false;
}; //end: PolicyModel (abstract class)


#endif
