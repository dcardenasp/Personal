
cmake_minimum_required(VERSION 2.8)

project(BayesianSegmentation)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(BayesianSegmentation MACOSX_BUNDLE BayesianSegmentation.cxx)
add_executable(membership MACOSX_BUNDLE MembershipFunctionEvaluation.cxx)
#add_executable(ParzenPDF MACOSX_BUNDLE ParzenPDF.cxx)
#add_executable(impainting MACOSX_BUNDLE impainting.cxx)

if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(BayesianSegmentation ITKReview ${ITK_LIBRARIES})
  target_link_libraries(membership ITKReview ${ITK_LIBRARIES})
#  target_link_libraries(ParzenPDF ITKReview ${ITK_LIBRARIES})
#  target_link_libraries(impainting ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(BayesianSegmentation ${ITK_LIBRARIES})
  target_link_libraries(membership ${ITK_LIBRARIES})
#  target_link_libraries(ParzenPDF ${ITK_LIBRARIES})
#  target_link_libraries(impainting ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )

