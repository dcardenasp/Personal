
cmake_minimum_required(VERSION 2.8)

project(BayesianSegmentation)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(ibs MACOSX_BUNDLE IterativeBayesianSegmentation.cxx)
add_executable(parzen MACOSX_BUNDLE MembershipFunctionEvaluation.cxx)
add_executable(prMapping MACOSX_BUNDLE PriorRegistration.cxx)
add_executable(imParzen MACOSX_BUNDLE ImageParzen.cxx)

if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(ibs ITKReview ${ITK_LIBRARIES})
  target_link_libraries(parzen ITKReview ${ITK_LIBRARIES})
  target_link_libraries(prMapping ITKReview ${ITK_LIBRARIES})
  target_link_libraries(imParzen ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(ibs ${ITK_LIBRARIES})
  target_link_libraries(parzen ${ITK_LIBRARIES})
  target_link_libraries(prMapping ${ITK_LIBRARIES})
  target_link_libraries(imParzen ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )
