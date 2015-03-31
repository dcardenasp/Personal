// image storage and I/O classes
#include "itkSize.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"

#include "itkMinimumMaximumImageCalculator.h"

#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBayesianClassifierImageFilter.h"

#include "itkImageRandomNonRepeatingConstIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"

#include "itkImageToListSampleFilter.h"

//#include "itkMembershipFunctionContainerType.h"
#include "itkWeightedCovarianceSampleFilter.h"
#include "itkWeightedParzenMembershipFunction.h"

int main(int argc, char *argv[])
{
  const unsigned int Dimension = 3;
  typedef float ImagePixelType;
  typedef itk::Vector< ImagePixelType, 1 > MeasurementVectorType;
  typedef itk::Image< ImagePixelType, Dimension > ImageType;
  typedef itk::Image< MeasurementVectorType, Dimension > ArrayImageType;
  typedef itk::VectorImage< ImagePixelType, Dimension > VectorImageType;

  unsigned int numberOfSamples = 100;
  float cov = 20;

  if( argc < 3 )
    {
    std::cerr << "Usage: " << argv[0] << " InputImage PriorVectorImage [ParzenSamples=100] [sigma=20]" << std::endl;
    return EXIT_FAILURE;
    }

  if(argc>3)
  {
    numberOfSamples=atoi(argv[3]);
  }

  if(argc>4)
    cov = atof(argv[4]);

  //
  // Load query, template and prior images.
  //
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer queryReader = ReaderType::New();
  queryReader->SetFileName( argv[1] );
  typedef itk::ImageFileReader< VectorImageType > VectorReaderType;
  VectorReaderType::Pointer priorReader = VectorReaderType::New();
  priorReader->SetFileName( argv[2] );
  try
  {
    queryReader->Update();
    priorReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception thrown " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }
  const unsigned int numberOfClasses = priorReader->GetOutput()->GetNumberOfComponentsPerPixel();

  //
  //
  //
  typedef itk::MinimumMaximumImageCalculator <ImageType>
          ImageCalculatorFilterType; 
  ImageCalculatorFilterType::Pointer imageCalculatorFilter
          = ImageCalculatorFilterType::New ();
  imageCalculatorFilter->SetImage(queryReader->GetOutput());
  imageCalculatorFilter->Compute();
  ImageType::PixelType vMax = imageCalculatorFilter->GetMaximum();
  ImageType::PixelType vMin = imageCalculatorFilter->GetMinimum();

  //
  // Computing mask for all upcoming estimations.
  //
  typedef unsigned short  LabelType;
  typedef itk::BayesianClassifierImageFilter< VectorImageType,LabelType,
          float,float >   ClassifierFilterType;
  ClassifierFilterType::Pointer classifier1 = ClassifierFilterType::New();
  classifier1->SetInput( priorReader->GetOutput() );
  typedef ClassifierFilterType::OutputImageType      ClassifierOutputImageType;
  typedef itk::BinaryThresholdImageFilter<ClassifierOutputImageType, ClassifierOutputImageType>
      BinaryThresholdImageFilterType;
  BinaryThresholdImageFilterType::Pointer threshold1
          = BinaryThresholdImageFilterType::New();
  threshold1->SetInput(classifier1->GetOutput());
  threshold1->SetLowerThreshold(1);
  threshold1->SetUpperThreshold(numberOfClasses);
  threshold1->SetInsideValue(1);
  threshold1->SetOutsideValue(0);
  typedef itk::BinaryBallStructuringElement<
          ClassifierOutputImageType::PixelType,Dimension> StructuringElementType;
  StructuringElementType structuringElement;
  structuringElement.SetRadius(10);
  structuringElement.CreateStructuringElement();
  typedef itk::BinaryDilateImageFilter <ClassifierOutputImageType,
          ClassifierOutputImageType, StructuringElementType> BinaryDilateImageFilterType;
  BinaryDilateImageFilterType::Pointer dilate1 = BinaryDilateImageFilterType::New();
  dilate1->SetInput(threshold1->GetOutput());
  dilate1->SetDilateValue(1);
  dilate1->SetKernel(structuringElement);
  typedef itk::ImageFileWriter< ClassifierOutputImageType >    WriterType;
  WriterType::Pointer writer1 = WriterType::New();
  writer1->SetFileName( "seg_mask.nii" );
  writer1->SetInput( dilate1->GetOutput() );
  writer1->Update();

  itk::ImageRegionConstIteratorWithIndex<VectorImageType> b(priorReader->GetOutput(), priorReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionConstIteratorWithIndex<ClassifierOutputImageType> Omega(dilate1->GetOutput(), dilate1->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<ImageType> x(queryReader->GetOutput(), queryReader->GetOutput()->GetLargestPossibleRegion());

  unsigned int OmegaSize = 0;
  for(Omega.GoToBegin(); !Omega.IsAtEnd(); ++Omega)
    if(Omega.Get()>0)
      OmegaSize++;

  //
  // Casting image to sample list
  //
  typedef itk::ComposeImageFilter< ImageType, ArrayImageType > CasterType;
  CasterType::Pointer caster = CasterType::New();
  caster->SetInput( queryReader->GetOutput() );
  caster->Update();
  typedef itk::Statistics::ImageToListSampleFilter< ArrayImageType,
          ClassifierOutputImageType > SampleType;

  //Parzen stuff:
  itk::ImageRegionConstIteratorWithIndex<VectorImageType> itIndVec(
          priorReader->GetOutput(), priorReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRandomNonRepeatingConstIteratorWithIndex<ArrayImageType>itRndImg(
          caster->GetOutput(), caster->GetOutput()->GetLargestPossibleRegion());
  itRndImg.SetNumberOfSamples( OmegaSize );
  typedef itk::Statistics::WeightedParzenMembershipFunction< MeasurementVectorType >
          WParzenMembershipFunctionType;
  WParzenMembershipFunctionType::Pointer wpmf = WParzenMembershipFunctionType::New();
  std::vector<WParzenMembershipFunctionType::WeightArrayType> wpmfWeights(numberOfClasses);

  typedef itk::Statistics::WeightedCovarianceSampleFilter< SampleType::ListSampleType >
          WeightedCovarianceAlgorithmType;
  WeightedCovarianceAlgorithmType::WeightArrayType weights;
  weights.SetSize(OmegaSize);
  std::vector<WParzenMembershipFunctionType::CovarianceMatrixType> wpmfCov(numberOfClasses);
  for (unsigned int c=0; c<numberOfClasses; c++)
  {
    (wpmfWeights[c]).SetSize(numberOfSamples);
    wpmfCov[c] = wpmf->GetCovariance();
    (wpmfCov[c])[0][0] = std::pow(cov,2.0);
  }

  std::vector<ArrayImageType::PixelType> samplelist(numberOfSamples);
  itRndImg.GoToBegin();
  Omega.GoToBegin();
  unsigned int r=0;
  while(!itRndImg.IsAtEnd())
  {
    Omega.SetIndex(itRndImg.GetIndex());
    if(Omega.Get()>0)
    {
      samplelist[r] = itRndImg.Get();
      b.SetIndex( itRndImg.GetIndex() );
      VectorImageType::PixelType br = b.Get();
      for(unsigned int c=0; c<numberOfClasses; c++)
      {
        (wpmfWeights[c]).SetElement(r,br[c]);
      }
      r++;
    }
    ++itRndImg;
    if(r >= numberOfSamples)
      break;
  }

  itk::Vector< float, 1 > meas;
  for (unsigned int c=0; c<numberOfClasses; c++)
  {
    wpmf->SetSampleList( samplelist );
    wpmf->SetWeights(wpmfWeights[c]);
    wpmf->SetCovariance(wpmfCov[c]);
    
    for (float v=vMin; v<=vMax; v+=(vMax-vMin)/1000)
    {
      meas[0] = v;
      std::cout << v << "\t" << wpmf->Evaluate(meas) << std::endl;
    }
  }
}
