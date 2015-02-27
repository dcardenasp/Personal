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

#include "itkBayesianClassifierInitializationImageFilter.h"

#include "itkRescaleIntensityImageFilter.h"

#include "itkWeightedMeanSampleFilter.h"
#include "itkWeightedCovarianceSampleFilter.h"
#include "itkGaussianMembershipFunction.h"
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
  gdOptimizer->SetNumberOfIterations( 2000 );
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


  //
  // Prior rigid mapping
  //
  const RigidTransformType::ParametersType finalRigidParameters =
          rigidRegistration->GetOutput()->Get()->GetParameters();
  RigidTransformType::Pointer finalRigidTransform = RigidTransformType::New();
  finalRigidTransform->SetFixedParameters( rigidRegistration->GetOutput()->Get()->GetFixedParameters() );
  finalRigidTransform->SetParameters( finalRigidParameters );
  VectorImageType::PixelType defaultPrior;
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
  vectorResampleFilter->Update();

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
  typedef itk::Statistics::WeightedMeanSampleFilter< SampleType >
          WeightedMeanAlgorithmType;

  WeightedMeanAlgorithmType::WeightArrayType weightArray( sample->Size() );
  weightArray.Fill( 1.0 );
  WeightedMeanAlgorithmType::Pointer weightedMeanAlgorithm =
          WeightedMeanAlgorithmType::New();
  weightedMeanAlgorithm->SetInput( sample );
  weightedMeanAlgorithm->SetWeights( weightArray );
  weightedMeanAlgorithm->Update();
  std::cout << "Sample weighted mean = "
            << weightedMeanAlgorithm->GetMean() << std::endl;

  typedef itk::Statistics::WeightedCovarianceSampleFilter< SampleType >
          WeightedCovarianceAlgorithmType;
  WeightedCovarianceAlgorithmType::Pointer weightedCovarianceAlgorithm =
                                        WeightedCovarianceAlgorithmType::New();
  weightedCovarianceAlgorithm->SetInput( sample );
  weightedCovarianceAlgorithm->SetWeights( weightArray );
  weightedCovarianceAlgorithm->Update();
  std::cout << "Sample weighted covariance = " << std::endl;
  std::cout << weightedCovarianceAlgorithm->GetCovarianceMatrix() << std::endl;


  typedef itk::Statistics::GaussianMembershipFunction< MeasurementVectorType >
          DensityFunctionType;
  DensityFunctionType::Pointer df1 = DensityFunctionType::New();
  df1->SetMeasurementVectorSize( 1 );
  DensityFunctionType::MeanVectorType mean
          = weightedMeanAlgorithm->GetMean();
  DensityFunctionType::CovarianceMatrixType cov
          = weightedCovarianceAlgorithm->GetCovarianceMatrix();
  df1->SetMean( mean );
  df1->SetCovariance( cov );

  typedef itk::BayesianClassifierInitializationImageFilter< ImageType >
          BayesianInitializerType;
  BayesianInitializerType::Pointer bayesianInitializer
          = BayesianInitializerType::New();

  bayesianInitializer->SetInput( queryReader->GetOutput() );
  bayesianInitializer->SetNumberOfClasses( atoi( argv[3] ) );
  // TODO add test where we specify membership functions
  typedef itk::ImageFileWriter<
    BayesianInitializerType::OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( bayesianInitializer->GetOutput() );
  writer->SetFileName( argv[2] );
  try
    {
    bayesianInitializer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "Exception thrown " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }
  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "Exception thrown " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }
  if( argv[4] && argv[5] )
    {
    typedef BayesianInitializerType::OutputImageType MembershipImageType;
    typedef itk::Image< MembershipImageType::InternalPixelType,
                        Dimension > ExtractedComponentImageType;
    ExtractedComponentImageType::Pointer extractedComponentImage =
                                    ExtractedComponentImageType::New();
    extractedComponentImage->CopyInformation(
                          bayesianInitializer->GetOutput() );
    extractedComponentImage->SetBufferedRegion( bayesianInitializer->GetOutput()->GetBufferedRegion() );
    extractedComponentImage->SetRequestedRegion( bayesianInitializer->GetOutput()->GetRequestedRegion() );
    extractedComponentImage->Allocate();
    typedef itk::ImageRegionConstIterator< MembershipImageType > ConstIteratorType;
    typedef itk::ImageRegionIterator< ExtractedComponentImageType > IteratorType;
    ConstIteratorType cit( bayesianInitializer->GetOutput(),
                     bayesianInitializer->GetOutput()->GetBufferedRegion() );
    IteratorType it( extractedComponentImage,
                     extractedComponentImage->GetLargestPossibleRegion() );
    const unsigned int componentToExtract = atoi( argv[4] );
    cit.GoToBegin();
    it.GoToBegin();
    SampleType::Iterator iter = sample->Begin() ;
    while( !cit.IsAtEnd() )
    {
      //it.Set(cit.Get()[componentToExtract]);
      it.Set( df1->Evaluate( iter.GetMeasurementVector() ) );
      ++it;
      ++cit;
      ++iter;
    }
    // Write out the rescaled extracted component
    typedef itk::Image< unsigned char, Dimension > OutputImageType;
    typedef itk::RescaleIntensityImageFilter<
      ExtractedComponentImageType, OutputImageType > RescalerType;
    RescalerType::Pointer rescaler = RescalerType::New();
    rescaler->SetInput( extractedComponentImage );
    rescaler->SetOutputMinimum( 0 );
    rescaler->SetOutputMaximum( 255 );
    typedef itk::ImageFileWriter<  OutputImageType
                        >  ExtractedComponentWriterType;
    ExtractedComponentWriterType::Pointer
               rescaledImageWriter = ExtractedComponentWriterType::New();
    rescaledImageWriter->SetInput( rescaler->GetOutput() );
    rescaledImageWriter->SetFileName( argv[5] );
    rescaledImageWriter->Update();
    }
  return EXIT_SUCCESS;
}
