
cmake_minimum_required(VERSION 2.8)

project(FeatureExtraction)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(mriPatchFeatures MACOSX_BUNDLE PatchFeatures.cxx)
add_executable(mriConvolutionFeatures MACOSX_BUNDLE ConvolutionFeatures.cxx)

if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(mriPatchFeatures ITKReview ${ITK_LIBRARIES})
  target_link_libraries(mriConvolutionFeatures ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(mriPatchFeatures ${ITK_LIBRARIES})
  target_link_libraries(mriConvolutionFeatures ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )
