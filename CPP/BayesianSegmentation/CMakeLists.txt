
cmake_minimum_required(VERSION 2.8)

project(BayesianSegmentation)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(BayesianSegmentation MACOSX_BUNDLE BayesianSegmentation.cxx)
add_executable(membership MACOSX_BUNDLE MembershipFunctionEvaluation.cxx)
#add_executable(gaussSegment MACOSX_BUNDLE ImageGaussianClassifier.cxx)


if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(BayesianSegmentation ITKReview ${ITK_LIBRARIES})
  target_link_libraries(membership ITKReview ${ITK_LIBRARIES})
#  target_link_libraries(gaussSegment ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(BayesianSegmentation ${ITK_LIBRARIES})
  target_link_libraries(membership ${ITK_LIBRARIES})
#  target_link_libraries(gaussSegment ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )
