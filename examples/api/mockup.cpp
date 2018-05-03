// #####
// ##### Application.h

RAJA_INLINE
int getApolloPolicyChoice(Apollo::Region *loop) 
{
    int choice = 0;

    // Here we refer to a stateful element of the Apollo
    // runtime that is doing learning strategies or, having
    // learned the best policy, returning the index.

    // Without further source instrumentation, this
    // block is effectively a loop.iterate() call.

    if (loop.doneLearning) {
        choice = loop.targetPolicyIndex;
    } else {
        choice = loop.getNextTestPolicy();
    }

    return choice;
}


// STEP 1 of 3: Set up a list of concrete policies for
//              each code region/class with policies that might
//              differ at runtime.
using Policy_XYZ_Seq = RAJA::nested::Policy<
	RAJA::nested::TypedFor<0, RAJA::loop_exec, Moment>,
	RAJA::nested::TypedFor<1, RAJA::loop_exec, Direction>,
	RAJA::nested::TypedFor<2, RAJA::loop_exec, Group>,
	RAJA::nested::TypedFor<3, RAJA::loop_exec, Zone>
	>;
using Policy_XYZ_Omp = RAJA::nested::Policy<
	RAJA::nested::TypedFor<2, RAJA::omp_parallel_for_exec, Group>,
	RAJA::nested::TypedFor<0, RAJA::loop_exec, Moment>,
	RAJA::nested::TypedFor<1, RAJA::loop_exec, Direction>,
	RAJA::nested::TypedFor<3, RAJA::loop_exec, Zone>
	>;
  	// ...
    // ... (policies continue)

// STEP 2 of 3: Make a template that uses some choice
//              to switch amongst the concrete policies:
template <typename BODY>
void XYZPolicySwitcher(int choice, BODY body) {
    switch (choice) {
    case 1: body(Policy_XYZ_Omp{}); break;
    case 2: body(Policy_XYZ_Omp2{}); break;
    case 3: body(Policy_XYZ_Omp3{}); break;
    case 4: body(Policy_XYZ_Omp4{}); break;
    case 0: 
    default: body(Policy_XYZ_Seq{}); break;
    }
}

// STEP 3 of 3: Use that template to wrap your loop
//              instead of referring to RAJA's classes
//              directly:
// #####
// ##### Application.cpp

#include "Apollo.h"

Apollo *apollo = new Apollo();
Apollo::Region *loop = new Apollo::Region(apollo);

loop.meta("Description", "Some basic description");
loop.meta("Type", "RAJA::nested::forall")

// NOTE: See section on lambdas' below!

loop.enter();
Application::XYZPolicySwitcher(
	Application::getApolloPolicyChoice(loop), 
	[=] (auto exec_policy) {
		RAJA::nested::forall(
			exec_policy,
			camp::make_tuple(
				RAJA::RangeSegment(0, some_things),
				RAJA::RangeSegment(0, more_things),
				RAJA::RangeSegment(0, diff_things),
				RAJA::RangeSegment(0, lost_things) ),
			APP_LAMBDA (Some so, More mo, Diff df, Lost lo) {
				// Computation here
			    // ...
            }
		);
	}
);
loop.leave();

// -----

// Section concerning use of lambdas, now or in the future:

// Idea:
// Express behaviors as lambdas.
// 
// Everything shown below has obvious and reasonable defaults, and
// in most cases would not need to be set, except for the block
// that is capturing named features.

// Reasoning:
// Some applications are generating output and changing their
// state during the test runs, and we might need to reset things
// once the learning phase is done. A lambda model like this
// gives us the hooks to do so.
//
// It also might be cleaner to say what features we want to capture
// without having to make changes to code inside the RAJA loop,
// supposing C++ scope rules allow us to reference such things.
// That's over my head at the moment.

// Example lambdas:
loop.on_enter({ loop.getGuidance(); });
loop.on_iteration({});
loop.on_leave({});

// Guidance mode consists of one or more tests.
loop.on_guidance_enter({
    loop.addTarget(apollo.FIND_LOWEST_TIME);
});
loop.on_guidance_test_enter({});
loop.on_guidance_test_iteration({
    loop.note("FeatureX", X);
    loop.note("FeatureY", Y);
    loop.note("FeatureZ", Z);
});
loop.on_guidance_test_leave({
    loop.sendNotes();
});
loop.on_guidance_leave({});