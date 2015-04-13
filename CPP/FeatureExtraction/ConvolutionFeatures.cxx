// image storage and I/O classes
#include "itkSize.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <numeric>

#include "itkConvolutionImageFilter.h"

const unsigned int Dimension = 3;
typedef float ImagePixelType;
typedef itk::Image< ImagePixelType, Dimension > ImageType;

void LoadSingleKernel(ImageType::Pointer kernel, std::string line )
{
  std::stringstream stream(line);
  std::vector<float> values(
     (std::istream_iterator<float>(stream)), // begin
     (std::istream_iterator<float>()));
  ImageType::IndexType start;
  start.Fill(0);
  ImageType::SizeType size;
  unsigned int dim=0;
  while(dim < Dimension)
  {
    size[dim] = static_cast<int>(values[dim]);
    dim++;
  }
  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);
  kernel->SetRegions(region);
  kernel->Allocate();
  itk::ImageRegionIterator<ImageType> imageIterator(kernel, region);
  imageIterator.GoToBegin();
  while(!imageIterator.IsAtEnd())
  {
    imageIterator.Set(values[dim]);
    ++imageIterator;
    dim++;
  }
}


int main(int argc, char *argv[])
{  
  typedef itk::Vector< ImagePixelType, 1 > MeasurementVectorType;
  typedef itk::Image< MeasurementVectorType, Dimension > ArrayImageType;
  typedef itk::VectorImage< ImagePixelType, Dimension > VectorImageType;

  if( argc < 4 )
  {
    std::cerr << "Usage: " << argv[0] << " InputImage filtersFile OutputVectorImage" << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Load image
  //
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer queryReader = ReaderType::New();
  queryReader->SetFileName( argv[1] );  
  try
  {
    queryReader->Update();    
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception thrown " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Filtering
  //
  unsigned int numberOfFilters=0;
  typedef itk::ConvolutionImageFilter<ImageType> FilterType;
  std::vector<ImageType::Pointer> FilteredImages;
  FilterType::Pointer convolutionFilter = FilterType::New();
  convolutionFilter->SetInput(queryReader->GetOutput());
  ImageType::Pointer kernel = ImageType::New();

  std::ifstream filterFile(argv[2]);
  std::string line;
  std::getline (filterFile,line);
  while(!line.empty())
  {
    LoadSingleKernel(kernel,line);
    std::getline (filterFile,line);
#if ITK_VERSION_MAJOR >= 4
    convolutionFilter->SetKernelImage(kernel);
#else
    convolutionFilter->SetImageKernelInput(kernel);
#endif
    convolutionFilter->Update();
    ImageType::Pointer output = convolutionFilter->GetOutput();
    output->DisconnectPipeline();
    FilteredImages.push_back( output );
    numberOfFilters++;
  }

  //
  // Collect filter outputs
  //
  typedef itk::ComposeImageFilter<ImageType> ComposeImageFilterType;
  ComposeImageFilterType::Pointer composeFilter = ComposeImageFilterType::New();
  for(unsigned int f=0; f<numberOfFilters; f++)
  {
    composeFilter->SetInput(f,FilteredImages[f]);
  }
  composeFilter->Update();

  //
  //Write output
  //
  typedef itk::ImageFileWriter< ComposeImageFilterType::OutputImageType >    WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[3] );
  writer->SetInput( composeFilter->GetOutput() );
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
  std::cerr << "Exception caught: " << std::endl;
  std::cerr << excp << std::endl;
  return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
