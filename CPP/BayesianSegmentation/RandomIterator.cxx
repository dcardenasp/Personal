//Basics
#include <cmath>
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageRandomNonRepeatingConstIteratorWithIndex.h"
//IO
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

int main(int argc, char *argv[])
{
  const unsigned int Dimension = 3;
  typedef float ImagePixelType;
  typedef itk::Vector< ImagePixelType, 1 > MeasurementVectorType;
  typedef itk::Image< ImagePixelType, Dimension > ImageType;
  typedef itk::Image< MeasurementVectorType, Dimension > ArrayImageType;
  typedef itk::VectorImage< ImagePixelType, Dimension > VectorImageType;

  if( argc < 4 )
    {
    std::cerr << "Usage: " << argv[0] << " ScalarImage VectorImage numberOfSamples" << std::endl;
    return EXIT_FAILURE;
    }

  //
  // Load
  //
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer templateReader = ReaderType::New();
  templateReader->SetFileName( argv[1] );
  typedef itk::ImageFileReader< VectorImageType > VectorReaderType;
  VectorReaderType::Pointer priorReader = VectorReaderType::New();
  priorReader->SetFileName( argv[2] );
  try
  {
    templateReader->Update();
    priorReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception thrown " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  unsigned int numberOfSamples = atoi(argv[3]);

  itk::ImageRandomNonRepeatingConstIteratorWithIndex<ArrayImageType>
              itRndImg(caster->GetOutput(), caster->GetOutput()->GetLargestPossibleRegion());
  itRndImg.SetNumberOfSamples( numberOfSamples );
  itRndImg.GoToBegin();
  unsigned int r=0;
  while(!itRndImg.IsAtEnd())
  {
    samplelist[r] = itRndImg.Get();
    itIndVec.SetIndex(itRndImg.GetIndex());
    for(unsigned int c=0; c<numberOfClasses; c++)
    {
      (wpmfWeights[c]).SetElement(r,itIndVec.Get()[c]);
    }
    ++itRndImg;
    r++;
  }
  return EXIT_SUCCESS;
}
