//Basics
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageRegionConstIterator.h"
#include "itkComposeImageFilter.h"
//IO
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
//Registration
#include "itkImageRegistrationMethodv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkVersorRigid3DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkResampleImageFilter.h"

#include "itkImageRegionConstIterator.h"

#include "itkBayesianClassifierInitializationImageFilter.h"
#include "itkBayesianClassifierImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkWeightedMeanSampleFilter.h"
#include "itkWeightedCovarianceSampleFilter.h"
#include "itkGaussianMembershipFunction.h"
#include "itkSampleClassifierFilter.h"
int main(int argc, char *argv[])
{
  const unsigned int Dimension = 3;
  typedef float ImagePixelType;
  typedef itk::Vector< ImagePixelType, 1 > MeasurementVectorType;
  typedef itk::Image< ImagePixelType, Dimension > ImageType;
  typedef itk::Image< MeasurementVectorType, Dimension > ArrayImageType;
  typedef itk::VectorImage< ImagePixelType, Dimension > VectorImageType;

  if( argc < 5 )
    {
    std::cerr << "Usage: " << argv[0] << " InputImage TemplateImage PriorVectorImage OutputImage" << std::endl;
    return EXIT_FAILURE;
    }

  //
  // Load query, template and prior images.
  //
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer queryReader = ReaderType::New();
  ReaderType::Pointer templateReader = ReaderType::New();
  queryReader->SetFileName( argv[1] );
  templateReader->SetFileName( argv[2] );
  typedef itk::ImageFileReader< VectorImageType > VectorReaderType;
  VectorReaderType::Pointer priorReader = VectorReaderType::New();
  priorReader->SetFileName( argv[3] );
  try
  {
    queryReader->Update();
    templateReader->Update();
    priorReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception thrown " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Template/Prior - Query Rigid Registration
  //
  std::cout << "Starting rigid registration" << std::endl;
  typedef itk::VersorRigid3DTransform< double >                 RigidTransformType;
  RigidTransformType::Pointer  initialTransform = RigidTransformType::New();
  typedef itk::CenteredTransformInitializer
          < RigidTransformType, ImageType, ImageType >          TransformInitializerType;
  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  initializer->SetTransform(   initialTransform );
  initializer->SetFixedImage(  queryReader->GetOutput() );
  initializer->SetMovingImage( templateReader->GetOutput() );
  initializer->MomentsOn();
  initializer->InitializeTransform();

  typedef itk::RegularStepGradientDescentOptimizerv4<double>    GDOptimizerType;
  GDOptimizerType::Pointer gdOptimizer = GDOptimizerType::New();
  typedef GDOptimizerType::ScalesType       OptimizerScalesType;
  OptimizerScalesType optimizerScales( initialTransform->GetNumberOfParameters() );
  const double translationScale = 1.0 / 1000.0;
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;
  optimizerScales[5] = translationScale;
  gdOptimizer->SetScales( optimizerScales );
  gdOptimizer->SetNumberOfIterations( 500 );
  gdOptimizer->SetLearningRate( 0.2 );
  gdOptimizer->SetMinimumStepLength( 0.00001 );
  gdOptimizer->SetReturnBestParametersAndValue(true);

  typedef itk::MattesMutualInformationImageToImageMetricv4
          < ImageType, ImageType >                              MetricType;
  MetricType::Pointer metric = MetricType::New();

  typedef itk::ImageRegistrationMethodv4
          < ImageType, ImageType, RigidTransformType >          RigidRegistrationType;
  const unsigned int numberOfLevels = 1;
  RigidRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize( 1 );
  shrinkFactorsPerLevel[0] = 1;
  RigidRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize( 1 );
  smoothingSigmasPerLevel[0] = 0;
  RigidRegistrationType::Pointer rigidRegistration = RigidRegistrationType::New();
  rigidRegistration->SetMetric(             metric                      );
  rigidRegistration->SetOptimizer(          gdOptimizer                 );
  rigidRegistration->SetFixedImage(         queryReader->GetOutput()    );
  rigidRegistration->SetMovingImage(        templateReader->GetOutput() );
  rigidRegistration->SetInitialTransform(   initialTransform            );
  rigidRegistration->SetNumberOfLevels( numberOfLevels );
  rigidRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
  rigidRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
  try
  {
    rigidRegistration->Update();
    std::cout << "Rigid Registration stop condition: "
              << rigidRegistration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }
  const unsigned int numberOfClasses = priorReader->GetOutput()->GetNumberOfComponentsPerPixel();

  //
  // Prior rigid mapping
  //
  std::cout << "Starting the prior mapping" << std::endl;
  const RigidTransformType::ParametersType finalRigidParameters =
          rigidRegistration->GetOutput()->Get()->GetParameters();
  RigidTransformType::Pointer finalRigidTransform = RigidTransformType::New();
  finalRigidTransform->SetFixedParameters( rigidRegistration->GetOutput()->Get()->GetFixedParameters() );
  finalRigidTransform->SetParameters( finalRigidParameters );
  VectorImageType::PixelType defaultPrior(numberOfClasses);
  defaultPrior.Fill(0.0); defaultPrior[0] = 1.0;
  typedef itk::ResampleImageFilter< VectorImageType, VectorImageType > VectorResampleFilterType;
  VectorResampleFilterType::Pointer vectorResampleFilter = VectorResampleFilterType::New();
  vectorResampleFilter->SetTransform( finalRigidTransform );
  vectorResampleFilter->SetInput( priorReader->GetOutput() );
  vectorResampleFilter->SetSize(    queryReader->GetOutput()->GetLargestPossibleRegion().GetSize() );
  vectorResampleFilter->SetOutputOrigin(  queryReader->GetOutput()->GetOrigin() );
  vectorResampleFilter->SetOutputSpacing( queryReader->GetOutput()->GetSpacing() );
  vectorResampleFilter->SetOutputDirection( queryReader->GetOutput()->GetDirection() );
  vectorResampleFilter->SetDefaultPixelValue( defaultPrior );
  try
  {
    vectorResampleFilter->Update();
    std::cout << "Priors mapped"
              << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Casting image to sample list
  //
  typedef itk::ComposeImageFilter< ImageType, ArrayImageType > CasterType;
  CasterType::Pointer caster = CasterType::New();
  caster->SetInput( queryReader->GetOutput() );
  caster->Update();
  typedef itk::Statistics::ImageToListSampleAdaptor< ArrayImageType > SampleType;
  SampleType::Pointer sample = SampleType::New();
  sample->SetImage( caster->GetOutput() );

  //
  // Parameter estimation
  //
  typedef itk::Statistics::ImageToListSampleAdaptor< ArrayImageType > SampleType;
  typedef itk::Statistics::WeightedCovarianceSampleFilter< SampleType >
          WeightedCovarianceAlgorithmType;
  std::vector< WeightedCovarianceAlgorithmType::Pointer > parameterEstimators;
  typedef itk::Statistics::GaussianMembershipFunction< MeasurementVectorType >
          MembershipFunctionType;
  typedef itk::BayesianClassifierInitializationImageFilter< ImageType >
          BayesianInitializerType;    
  typedef BayesianInitializerType::MembershipFunctionContainerType
          MembershipFunctionContainerType;
  MembershipFunctionContainerType::Pointer membershipFunctionContainer
          = MembershipFunctionContainerType::New();
  membershipFunctionContainer->Reserve(numberOfClasses);
  for ( unsigned int c = 0; c < numberOfClasses; ++c )
  {
    std::cout << "Class " << c << " :" << std::endl;
    WeightedCovarianceAlgorithmType::WeightArrayType weights;
    weights.SetSize(sample->Size());
    itk::ImageRegionConstIterator<VectorImageType> prIt(vectorResampleFilter->GetOutput(),vectorResampleFilter->GetOutput()->GetLargestPossibleRegion());
    prIt.GoToBegin();
    unsigned int r=0;
    while(!prIt.IsAtEnd())
    {
      VectorImageType::PixelType b = prIt.Get();
      weights[r] = b[c];
      ++prIt;
      r++;
    }
    parameterEstimators.push_back( WeightedCovarianceAlgorithmType::New() );
    parameterEstimators[c]->SetInput( sample );
    parameterEstimators[c]->SetWeights( weights );
    parameterEstimators[c]->Update();
    std::cout <<"Sample weighted mean = "
              << parameterEstimators[c]->GetMean() << std::endl;
    std::cout << "Sample weighted covariance = " << std::endl;
    std::cout << parameterEstimators[c]->GetCovarianceMatrix() << std::endl;
    MembershipFunctionType::Pointer membershipFunction =
              MembershipFunctionType::New();
    membershipFunction->SetMean( parameterEstimators[c]->GetMean() );
    membershipFunction->SetCovariance( parameterEstimators[c]->GetCovarianceMatrix() );
    membershipFunctionContainer->SetElement( c, membershipFunction.GetPointer() );
  }

  //
  // Compute class likelihoods
  //
  BayesianInitializerType::Pointer bayesianInitializer
          = BayesianInitializerType::New();
  bayesianInitializer->SetInput( queryReader->GetOutput() );
  bayesianInitializer->SetNumberOfClasses( numberOfClasses );
  bayesianInitializer->SetMembershipFunctions( membershipFunctionContainer );
  try
  {
    bayesianInitializer->Update();
    std::cout << "Class Likelihoods computed"
              << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Compute posteriors
  //
  typedef unsigned short  LabelType;
  typedef itk::BayesianClassifierImageFilter< VectorImageType,LabelType,
          float,float >   ClassifierFilterType;
  ClassifierFilterType::Pointer classifier = ClassifierFilterType::New();
  classifier->SetInput( bayesianInitializer->GetOutput() );
  classifier->SetPriors( vectorResampleFilter->GetOutput() );
  typedef ClassifierFilterType::OutputImageType      ClassifierOutputImageType;
  typedef itk::ImageFileWriter< ClassifierOutputImageType >    WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[4] );
  writer->SetInput( classifier->GetOutput() );
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
  // Testing print
  classifier->Print( std::cout );
  std::cout << "Test passed." << std::endl;

  return EXIT_SUCCESS;
}
