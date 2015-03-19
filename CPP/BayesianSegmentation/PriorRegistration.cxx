//Basics
#include <cmath>
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageRegionConstIterator.h"
#include "itkComposeImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkImageRandomNonRepeatingConstIteratorWithIndex.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
//IO
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
//Registration
#include "itkCompositeTransform.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkVersorRigid3DTransform.h"
#include "itkBSplineTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"
#include "itkResampleImageFilter.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"

#include "itkImageRegionConstIterator.h"

#include "itkBayesianClassifierInitializationImageFilter.h"
#include "itkBayesianClassifierImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkImageToListSampleFilter.h"
#include "itkWeightedMeanSampleFilter.h"
#include "itkWeightedCovarianceSampleFilter.h"
#include "itkGaussianMembershipFunction.h"
#include "itkWeightedParzenMembershipFunction.h"
#include "itkSampleClassifierFilter.h"
#include "itkGaussianDistribution.h"

//  The following section of code implements a Command observer
//  used to monitor the evolution of the registration process.
//
#include "itkCommand.h"

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
  typedef  RegistrationInterfaceCommand   Self;
  typedef  itk::Command                   Superclass;
  typedef  itk::SmartPointer<Self>        Pointer;
  itkNewMacro( Self );
protected:
  RegistrationInterfaceCommand() {};
public:
  typedef   TRegistration                          RegistrationType;
  // The Execute function simply calls another version of the \code{Execute()}
  // method accepting a \code{const} input object
  void Execute( itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
    {
    Execute( (const itk::Object *) object , event );
    }
  void Execute(const itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
    {
    if( !(itk::MultiResolutionIterationEvent().CheckEvent( &event ) ) )
      {
      return;
      }
    std::cout << "\nObserving from class " << object->GetNameOfClass();
    if (!object->GetObjectName().empty())
      {
      std::cout << " \"" << object->GetObjectName() << "\"" << std::endl;
      }
    const RegistrationType * registration = static_cast<const RegistrationType *>( object );
    if(registration == 0)
      {
      itkExceptionMacro(<< "Dynamic cast failed, object of type " << object->GetNameOfClass());
      }
    unsigned int currentLevel = registration->GetCurrentLevel();
    typename RegistrationType::ShrinkFactorsPerDimensionContainerType shrinkFactors =
                                              registration->GetShrinkFactorsPerDimension( currentLevel );
    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmas =
                                                            registration->GetSmoothingSigmasPerLevel();
    std::cout << "-------------------------------------" << std::endl;
    std::cout << " Current multi-resolution level = " << currentLevel << std::endl;
    std::cout << "    shrink factor = " << shrinkFactors << std::endl;
    std::cout << "    smoothing sigma = " << smoothingSigmas[currentLevel] << std::endl;
    std::cout << std::endl;
    }
};

class CommandIterationUpdate : public itk::Command
{
public:
  typedef CommandIterationUpdate   Self;
  typedef itk::Command             Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate() {};
public:
  typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
  typedef const OptimizerType*                               OptimizerPointer;
  void Execute(itk::Object *caller, const itk::EventObject & event) ITK_OVERRIDE
  {
    Execute( (const itk::Object *)caller, event);
  }
  void Execute(const itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
  {
    OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << " = ";
    std::cout << optimizer->GetValue() << " : ";
  //std::cout << optimizer->GetCurrentPosition() << std::endl;
    std::cout << std::endl;
  }
};



int main(int argc, char *argv[])
{
  const unsigned int Dimension = 3;
  typedef float ImagePixelType;
  typedef itk::Vector< ImagePixelType, 1 > MeasurementVectorType;
  typedef itk::Image< ImagePixelType, Dimension > ImageType;
  typedef itk::Image< MeasurementVectorType, Dimension > ArrayImageType;
  typedef itk::VectorImage< ImagePixelType, Dimension > VectorImageType;

  unsigned int memFunc = 0;
  double alpha = 1.0;
  float zero = 1e-4;
  unsigned int numberOfSamples = 100;

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
  // Template/Prior - Query Registration
  //

  std::cout << "Starting rigid registration" << std::endl;
  typedef itk::VersorRigid3DTransform< double >                 RigidTransformType;
  RigidTransformType::Pointer  initialTransform = RigidTransformType::New();
  RigidTransformType::Pointer  rigidTransform = RigidTransformType::New();
  typedef itk::CenteredTransformInitializer
          < RigidTransformType, ImageType, ImageType >          TransformInitializerType;
  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  initializer->SetTransform(   initialTransform );
  initializer->SetFixedImage(  queryReader->GetOutput() );
  initializer->SetMovingImage( templateReader->GetOutput() );
  initializer->MomentsOn();
  initializer->InitializeTransform();

  typedef itk::MattesMutualInformationImageToImageMetricv4
          < ImageType, ImageType >                              MetricType;
  MetricType::Pointer metric = MetricType::New();

  typedef itk::RegularStepGradientDescentOptimizerv4<double>    GDOptimizerType;
  typedef GDOptimizerType::ScalesType       OptimizerScalesType;
  OptimizerScalesType optimizerScales( initialTransform->GetNumberOfParameters() );
  const double translationScale = 1.0 / 1000.0;
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;
  optimizerScales[5] = translationScale;
  /*
  typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
  ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
  scalesEstimator->SetMetric( metric );
  scalesEstimator->SetTransformForward( true );
  scalesEstimator->SetSmallParameterVariation( 1.0 );*/
  GDOptimizerType::Pointer gdOptimizer = GDOptimizerType::New();
  gdOptimizer->SetNumberOfIterations( 100 );
  gdOptimizer->SetLearningRate( 0.2 );
  gdOptimizer->SetMinimumStepLength( 0.001 );
  gdOptimizer->SetReturnBestParametersAndValue(true);
  //gdOptimizer->SetScalesEstimator( scalesEstimator );
  gdOptimizer->SetScales( optimizerScales );
  gdOptimizer->SetRelaxationFactor( 0.5 );
  gdOptimizer->SetDoEstimateLearningRateOnce( true );
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  gdOptimizer->AddObserver( itk::IterationEvent(), observer );

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
  typedef itk::CompositeTransform< double, Dimension >  CompositeTransformType;
  CompositeTransformType::Pointer  compositeTransform  =
          CompositeTransformType::New();
  compositeTransform->AddTransform( initialTransform );
  compositeTransform->AddTransform( rigidRegistration->GetModifiableTransform() );
  rigidTransform->SetParameters( rigidRegistration->GetOutput()->Get()->GetParameters() );

  std::cout << "Starting affine registration" << std::endl;
  typedef itk::AffineTransform< double, Dimension >       AffineTransformType;
  AffineTransformType::Pointer  affineTransform = AffineTransformType::New();
  affineTransform->SetCenter( rigidTransform->GetCenter() );
  affineTransform->SetTranslation( rigidTransform->GetTranslation() );
  affineTransform->SetMatrix( rigidTransform->GetMatrix() );
  typedef itk::ConjugateGradientLineSearchOptimizerv4Template< double >
          AOptimizerType;
  typedef itk::ImageRegistrationMethodv4< ImageType, ImageType,
          AffineTransformType > ARegistrationType;
  AOptimizerType::Pointer      affineOptimizer     = AOptimizerType::New();
  //MetricType::Pointer          affineMetric        = MetricType::New();
  ARegistrationType::Pointer   affineRegistration  = ARegistrationType::New();
  affineRegistration->SetOptimizer(     affineOptimizer     );
  affineRegistration->SetMetric( metric  );
  affineRegistration->SetMovingInitialTransform(  compositeTransform  );
  affineRegistration->SetFixedImage( queryReader->GetOutput()    );
  affineRegistration->SetMovingImage( templateReader->GetOutput() );
  affineRegistration->SetInitialTransform( affineTransform );
  //affineRegistration->SetMovingInitialTransformInput( rigidRegistration->GetTransformOutput() );

  typedef itk::RegistrationParameterScalesFromPhysicalShift<
    MetricType> ScalesEstimatorType;
  ScalesEstimatorType::Pointer scalesEstimator =
    ScalesEstimatorType::New();
  scalesEstimator->SetMetric( metric );
  scalesEstimator->SetTransformForward( true );
  affineOptimizer->SetScalesEstimator( scalesEstimator );
  affineOptimizer->SetDoEstimateLearningRateOnce( true );
  affineOptimizer->SetDoEstimateLearningRateAtEachIteration( false );
  affineOptimizer->SetLowerLimit( 0 );
  affineOptimizer->SetUpperLimit( 2 );
  affineOptimizer->SetEpsilon( 0.2 );
  affineOptimizer->SetNumberOfIterations( 100 );
  affineOptimizer->SetMinimumConvergenceValue( 1e-6 );
  affineOptimizer->SetConvergenceWindowSize( 10 );
  CommandIterationUpdate::Pointer observer2 = CommandIterationUpdate::New();
  affineOptimizer->AddObserver( itk::IterationEvent(), observer2 );
  const unsigned int numberOfLevels2 = 1;
  ARegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel2;
  shrinkFactorsPerLevel2.SetSize( numberOfLevels2 );
  shrinkFactorsPerLevel2[0] = 2;
  shrinkFactorsPerLevel2[1] = 1;
  ARegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel2;
  smoothingSigmasPerLevel2.SetSize( numberOfLevels2 );
  smoothingSigmasPerLevel2[0] = 1;
  smoothingSigmasPerLevel2[1] = 0;
  affineRegistration->SetNumberOfLevels ( numberOfLevels2 );
  affineRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel2 );
  affineRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel2 );
  // Create the Command interface observer and register it with the optimizer.
  //
  typedef RegistrationInterfaceCommand<ARegistrationType> AffineCommandType;
  AffineCommandType::Pointer command2 = AffineCommandType::New();
  affineRegistration->AddObserver( itk::MultiResolutionIterationEvent(), command2 );
  try
  {
    affineRegistration->Update();
    std::cout << "Optimizer stop condition: "
              << affineRegistration->
                          GetOptimizer()->GetStopConditionDescription()
              << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return EXIT_FAILURE;
  }
  //std::cout << "Scales:  " << gdOptimizer->GetScales() << std::endl;

  //
  // Prior rigid mapping
  //
  std::cout << "Starting the prior mapping" << std::endl;
  compositeTransform->AddTransform( affineRegistration->GetModifiableTransform() );

  const unsigned int numberOfClasses = priorReader->GetOutput()->GetNumberOfComponentsPerPixel();
  /*const RigidTransformType::ParametersType finalRigidParameters =
          rigidRegistration->GetOutput()->Get()->GetParameters();
  RigidTransformType::Pointer finalRigidTransform = RigidTransformType::New();
  finalRigidTransform->SetFixedParameters( rigidRegistration->GetOutput()->Get()->GetFixedParameters() );
  finalRigidTransform->SetParameters( finalRigidParameters );*/
  VectorImageType::PixelType defaultPrior(numberOfClasses);
  defaultPrior.Fill(0.0); defaultPrior[0] = 1.0;
  typedef itk::ResampleImageFilter< VectorImageType, VectorImageType > VectorResampleFilterType;
  VectorResampleFilterType::Pointer vectorResampleFilter = VectorResampleFilterType::New();
  //vectorResampleFilter->SetTransform( finalRigidTransform );
  vectorResampleFilter->SetTransform( compositeTransform );
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

  itk::ImageRegionIterator<VectorImageType> b(vectorResampleFilter->GetOutput(), vectorResampleFilter->GetOutput()->GetLargestPossibleRegion());

  for(b.GoToBegin(); !b.IsAtEnd(); ++b)
  {
    float z=0;
    VectorImageType::PixelType br = b.Get();
    for (unsigned int c=0; c<numberOfClasses; c++)
    {
      z += br[c];
    }
    for (unsigned int c=0; c<numberOfClasses; c++)
    {
      br[c] = (br[c])/z;
    }
    b.Set(br);
  }
  typedef itk::ImageFileWriter< VectorImageType >    WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[4] );
  writer->SetInput( vectorResampleFilter->GetOutput() );
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
