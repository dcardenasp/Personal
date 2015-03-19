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
  bool registerPrior = true;

  if( argc < 5 )
    {
    std::cerr << "Usage: " << argv[0] << " InputImage TemplateImage PriorVectorImage OutputImage [memfunc] [alpha] [ParzenSamples] [registerPrior?]" << std::endl;
    return EXIT_FAILURE;
    }

  if(argc>5)
  {
    memFunc = atoi(argv[5]);
  }
  std::cout << "Class membership functions are ";
  switch(memFunc)
  {
  case 0:
      std::cout << "Gaussians" << std::endl;
      break;
  case 1:
      std::cout << "Parzen-based" << std::endl;
      break;
  }

  if(argc>6)
  {
    alpha = atof(argv[6]);
  }
  std::cout << "Alpha factor for entropy and divergence is " << alpha << std::endl;

  if(argc>7)
  {
    numberOfSamples = atoi(argv[7]);
  }
  std::cout << "Number of Samples for the parzen estimation is " << numberOfSamples << std::endl;

  if(argc>8)
  {
    if(atoi(argv[8])==0)
      registerPrior = false;
    else
      registerPrior = true;
  }
  if(registerPrior)
    std::cout << "Priors will be aligned" << std::endl;
  else
    std::cout << "Priors won't be aligned" << std::endl;


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
  VectorImageType::Pointer priorImage = VectorImageType::New();
  const unsigned int numberOfClasses = priorReader->GetOutput()->GetNumberOfComponentsPerPixel();
  if (registerPrior)
  {
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
  priorImage = vectorResampleFilter->GetOutput();
  }
  else
  {
    priorImage = priorReader->GetOutput();
  }

  //
  // Computing mask for all upcoming estimations.
  //
  typedef unsigned short  LabelType;
  typedef itk::BayesianClassifierImageFilter< VectorImageType,LabelType,
          float,float >   ClassifierFilterType;
  ClassifierFilterType::Pointer classifier1 = ClassifierFilterType::New();
  classifier1->SetInput( priorImage );
  typedef ClassifierFilterType::OutputImageType      ClassifierOutputImageType;
  typedef itk::BinaryThresholdImageFilter <ClassifierOutputImageType, ClassifierOutputImageType>
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
  BinaryDilateImageFilterType::Pointer dilate1
          = BinaryDilateImageFilterType::New();
  dilate1->SetInput(threshold1->GetOutput());
  dilate1->SetDilateValue(1);
  dilate1->SetKernel(structuringElement);
  typedef itk::ImageFileWriter< ClassifierOutputImageType >    WriterType;
  WriterType::Pointer writer1 = WriterType::New();
  writer1->SetFileName( "seg_mask.nii" );
  writer1->SetInput( dilate1->GetOutput() );
  writer1->Update();

  //
  // Casting image to sample list
  //
  typedef itk::ComposeImageFilter< ImageType, ArrayImageType > CasterType;
  CasterType::Pointer caster = CasterType::New();
  caster->SetInput( queryReader->GetOutput() );
  caster->Update();
  //typedef itk::Statistics::ImageToListSampleAdaptor< ArrayImageType > SampleType;
  typedef itk::Statistics::ImageToListSampleFilter< ArrayImageType,
          ClassifierOutputImageType > SampleType;
  SampleType::Pointer sample = SampleType::New();
  sample->SetInput( caster->GetOutput() );
  sample->SetMaskImage( dilate1->GetOutput() );
  sample->SetMaskValue( 1 );
  sample->Update();

  //
  // Set the posterior image
  //
  VectorImageType::Pointer posterior = VectorImageType::New();
  posterior->SetRegions( queryReader->GetOutput()->GetLargestPossibleRegion() );
  posterior->SetOrigin( queryReader->GetOutput()->GetOrigin() );
  posterior->SetDirection( queryReader->GetOutput()->GetDirection() );
  posterior->SetSpacing( queryReader->GetOutput()->GetSpacing() );
  posterior->SetVectorLength(numberOfClasses);
  posterior->Allocate();
  VectorImageType::PixelType tmp(numberOfClasses);
  tmp.Fill(1.0/float(numberOfClasses));
  posterior->FillBuffer(tmp);

  VectorImageType::Pointer hidden = VectorImageType::New();
  hidden->SetRegions( queryReader->GetOutput()->GetLargestPossibleRegion() );
  hidden->SetOrigin( queryReader->GetOutput()->GetOrigin() );
  hidden->SetDirection( queryReader->GetOutput()->GetDirection() );
  hidden->SetSpacing( queryReader->GetOutput()->GetSpacing() );
  hidden->SetVectorLength(numberOfClasses);
  hidden->Allocate();
  hidden->FillBuffer(tmp);

  VectorImageType::Pointer joint = VectorImageType::New();
  joint->SetRegions( queryReader->GetOutput()->GetLargestPossibleRegion() );
  joint->SetOrigin( queryReader->GetOutput()->GetOrigin() );
  joint->SetDirection( queryReader->GetOutput()->GetDirection() );
  joint->SetSpacing( queryReader->GetOutput()->GetSpacing() );
  joint->SetVectorLength(numberOfClasses);
  joint->Allocate();
  joint->FillBuffer(tmp);

  VectorImageType::Pointer conditional = VectorImageType::New();
  conditional->SetRegions( queryReader->GetOutput()->GetLargestPossibleRegion() );
  conditional->SetOrigin( queryReader->GetOutput()->GetOrigin() );
  conditional->SetDirection( queryReader->GetOutput()->GetDirection() );
  conditional->SetSpacing( queryReader->GetOutput()->GetSpacing() );
  conditional->SetVectorLength(numberOfClasses);
  conditional->Allocate();
  conditional->FillBuffer(tmp);

  itk::ImageRegionIterator<VectorImageType> q(hidden, hidden->GetLargestPossibleRegion());
  itk::ImageRegionIterator<VectorImageType> p(joint, joint->GetLargestPossibleRegion());
  itk::ImageRegionIterator<VectorImageType> pos(posterior, posterior->GetLargestPossibleRegion());
  itk::ImageRegionIterator<VectorImageType> f(conditional, conditional->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<VectorImageType> b(priorImage, priorImage->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<ClassifierOutputImageType> Omega(dilate1->GetOutput(), dilate1->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<ImageType> x(queryReader->GetOutput(), queryReader->GetOutput()->GetLargestPossibleRegion());

  unsigned int OmegaSize = 0;
  for(Omega.GoToBegin(); !Omega.IsAtEnd(); ++Omega)
    if(Omega.Get()>0)
      OmegaSize++;

  //
  // Main loop
  //
  typedef itk::BayesianClassifierInitializationImageFilter< ImageType >
          BayesianInitializerType;
  BayesianInitializerType::Pointer bayesianInitializer
          = BayesianInitializerType::New();
  unsigned int MaximumNumberOfIterations = 100;
  float cost_func = 0;
  for(unsigned int CurrentNumberOfIterations=0;
      CurrentNumberOfIterations < MaximumNumberOfIterations;
      CurrentNumberOfIterations++)
  {
    //std::cout << CurrentNumberOfIterations << "/" << MaximumNumberOfIterations << std::endl;
  //
  // Parameter estimation
  //
    //typedef itk::Statistics::ImageToListSampleAdaptor< ArrayImageType > SampleType;
    typedef itk::Statistics::WeightedCovarianceSampleFilter< SampleType::ListSampleType >
            WeightedCovarianceAlgorithmType;
    std::vector< WeightedCovarianceAlgorithmType::Pointer > parameterEstimators;
    typedef itk::Statistics::GaussianMembershipFunction< MeasurementVectorType >
            MembershipFunctionType;
    typedef BayesianInitializerType::MembershipFunctionContainerType
            MembershipFunctionContainerType;
    MembershipFunctionContainerType::Pointer membershipFunctionContainer
            = MembershipFunctionContainerType::New();
    membershipFunctionContainer->Reserve(numberOfClasses);

    double tmp1=0;
    switch(memFunc)
    {
    case 0:
      for ( unsigned int c = 0; c < numberOfClasses; ++c )
      {
        //std::cout << "Class " << c << " :" << std::endl;
        MembershipFunctionType::Pointer membershipFunction =
                MembershipFunctionType::New();
        WeightedCovarianceAlgorithmType::WeightArrayType weights;
        weights.SetSize(sample->GetOutput()->Size());
        q.GoToBegin();
        p.GoToBegin();
        b.GoToBegin();
        f.GoToBegin();
        pos.GoToBegin();
        Omega.GoToBegin();
        tmp1=0;
        unsigned int r=0;
        while(!q.IsAtEnd())
        {
          if(Omega.Get()>0)
          {
            VectorImageType::PixelType qr = q.Get();
            VectorImageType::PixelType br = b.Get();
            VectorImageType::PixelType fr = f.Get();
            double w = br[c]/(qr[c]+zero);
            w = std::pow(w,alpha);
            weights[r] = w*fr[c];
            r++;
          }
          ++q;
          ++p;
          ++b;
          ++f;
          ++Omega;
        }
        parameterEstimators.push_back( WeightedCovarianceAlgorithmType::New() );
        parameterEstimators[c]->SetInput( sample->GetOutput() );
        parameterEstimators[c]->SetWeights( weights );
        parameterEstimators[c]->Update();
        /*std::cout <<"Sample weighted mean = "
                  << parameterEstimators[c]->GetMean() << std::endl;
        std::cout << "Sample weighted covariance = " << std::endl;
        std::cout << parameterEstimators[c]->GetCovarianceMatrix() << std::endl;*/
        membershipFunction->SetMean( parameterEstimators[c]->GetMean() );
        membershipFunction->SetCovariance( parameterEstimators[c]->GetCovarianceMatrix() );
        membershipFunctionContainer->SetElement( c, membershipFunction.GetPointer() );
      }
      break;
    case 1:
      //std::cout << "Selecting samples for building the pdf" << std::endl;
      itk::ImageRegionConstIteratorWithIndex<VectorImageType> itIndVec(
              priorImage, priorImage->GetLargestPossibleRegion());
      itk::ImageRandomNonRepeatingConstIteratorWithIndex<ArrayImageType>
              itRndImg(caster->GetOutput(), caster->GetOutput()->GetLargestPossibleRegion());
      itRndImg.SetNumberOfSamples( OmegaSize );
      typedef itk::Statistics::WeightedParzenMembershipFunction< MeasurementVectorType >
              WParzenMembershipFunctionType;
      WParzenMembershipFunctionType::Pointer wpmf = WParzenMembershipFunctionType::New();
      std::vector<WParzenMembershipFunctionType::WeightArrayType> wpmfWeights(numberOfClasses);

      WeightedCovarianceAlgorithmType::WeightArrayType weights;
      weights.SetSize(OmegaSize);

      std::vector<WParzenMembershipFunctionType::CovarianceMatrixType> wpmfCov(numberOfClasses);

      for (unsigned int c=0; c<numberOfClasses; c++)
      {
        (wpmfWeights[c]).SetSize(numberOfSamples);
        wpmfCov[c] = wpmf->GetCovariance();
        (wpmfCov[c])[0][0] = std::pow(20.0,2.0);
      }

      std::vector<ArrayImageType::PixelType> samplelist(numberOfSamples);

      itRndImg.GoToBegin();
      Omega.GoToBegin();
      unsigned int r=0;
      while(!itRndImg.IsAtEnd())
      {
        if(Omega.Get()>0)
        {
          samplelist[r] = itRndImg.Get();
          std::cout << samplelist[r];
          pos.SetIndex( itRndImg.GetIndex() );
          VectorImageType::PixelType pos_r = pos.Get();
          for(unsigned int c=0; c<numberOfClasses; c++)
          {
            (wpmfWeights[c]).SetElement(r,pos_r[c]);
            std::cout << "\t" << (wpmfWeights[c]).GetElement(r);
          }
          std::cout << std::endl;
          r++;
        }
        ++itRndImg;
        ++Omega;
        if(r >= numberOfSamples)
          break;
      }
      //std::cout << "samples selected" << std::endl;

      //Covariance update
      q.GoToBegin();
      p.GoToBegin();
      b.GoToBegin();
      x.GoToBegin();
      pos.GoToBegin();
      Omega.GoToBegin();
      r=0;
      std::vector<double> wfd(numberOfClasses);
      std::vector<double> wf(numberOfClasses);
      for(unsigned int c=0; c<numberOfClasses; c++)
      {
        wfd[c] = 0;
        wf[c] = 0;
      }      
      while(!q.IsAtEnd())
      {
        if(Omega.Get()>0)
        {
          VectorImageType::PixelType qr = q.Get();
          VectorImageType::PixelType pr = p.Get();
          VectorImageType::PixelType br = b.Get();
          ImageType::PixelType xr = x.Get();
          std::vector<double> w(numberOfClasses);
          std::vector<double> fd(numberOfClasses);
          std::vector<double> nf(numberOfClasses);
          for(unsigned int c=0; c<numberOfClasses; c++)
          {
            fd[c] = 0;
            nf[c] = 0;
            w[c] = br[c]/(qr[c]+zero);
            w[c] = std::pow(w[c],alpha);
          }
          for(unsigned int m=0; m<numberOfSamples; m++)
          {
            float d = (x.Get()-samplelist[m][0]);
            d = std::pow(d,2.0);
            itk::Statistics::GaussianDistribution::Pointer normal =
                itk::Statistics::GaussianDistribution::New();
            normal->SetMean(samplelist[m][0]);
            for(unsigned int c=0; c<numberOfClasses; c++)
            {
              normal->SetVariance((wpmfCov[c])[0][0]);
              double k = normal->EvaluatePDF(x.Get());
              fd[c] = fd[c] + (wpmfWeights[c]).GetElement(m)*k*d;
              nf[c] = nf[c] + (wpmfWeights[c]).GetElement(m)*k;
            }
          }
          for(unsigned int c=0; c<numberOfClasses; c++)
          {
            wfd[c] = wfd[c] + w[c]*fd[c];
            wf[c] = wf[c] + w[c]*nf[c];
          }
          r++;
        }
        ++q;
        ++p;
        ++b;
        ++pos;
        ++Omega;
      }

      for(unsigned int c=0; c<numberOfClasses; c++)
      {
        (wpmfCov[c])[0][0] = wfd[c]/wf[c];
        std::cout << "Class " << c << ": " << wpmfCov[c] << std::endl;
      }

      for (unsigned int c=0; c<numberOfClasses; c++)
      {
        wpmf->SetSampleList( samplelist );
        wpmf->SetWeights(wpmfWeights[c]);
        wpmf->SetCovariance(wpmfCov[c]);
        membershipFunctionContainer->SetElement( c, wpmf.GetPointer() );
        std::cout << "class " << c << ": membership added" << std::endl;
      }
      break;
    }

    //std::cout << "conditional parameters computed" << std::endl;
/*
    for (unsigned int c=0; c<numberOfClasses; c++)
      std::cout << membershipFunctionContainer->GetElement(c) << std::endl;
*/
  //
  // Compute class likelihoods
  //
    bayesianInitializer->SetInput( queryReader->GetOutput() );
    bayesianInitializer->SetNumberOfClasses( numberOfClasses );
    bayesianInitializer->SetMembershipFunctions( membershipFunctionContainer );
    try
    {
      bayesianInitializer->Update();
      /*std::cout << "Class Likelihoods computed"
                << std::endl;*/
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
    {
      VectorImageType::Pointer conditional_temporal = bayesianInitializer->GetOutput();
      conditional_temporal->DisconnectPipeline();
      itk::ImageRegionConstIterator<VectorImageType> f1(conditional_temporal, conditional_temporal->GetLargestPossibleRegion());
      f.GoToBegin();
      f1.GoToBegin();
      while(!f.IsAtEnd())
      {
        f.Set(f1.Get());
        ++f;
        ++f1;
      }
    }

    f.GoToBegin();
    b.GoToBegin();
    q.GoToBegin();
    pos.GoToBegin();
    Omega.GoToBegin();
    unsigned int r=0;
    float beta = 1.0;
    if(alpha!=0.0)
    {
      beta = (alpha-1.0)/alpha;
    }
    float pr=0;
    cost_func = 0;
    float pos_change=0;
    while(!q.IsAtEnd())
    {
      if (Omega.Get()>0)
      {
        VectorImageType::PixelType qr = q.Get();
        VectorImageType::PixelType br = b.Get();
        VectorImageType::PixelType fr = f.Get();
        VectorImageType::PixelType pos_r = pos.Get();
        std::vector<float> pos_ant(numberOfClasses);
        pr=0;
        for(unsigned int c=0; c<numberOfClasses; c++)
        {
          pos_ant[c] = pos_r[c];
          pos_r[c] = (br[c])*(fr[c]);
          pr += pos_r[c];
        }
        if(pr<1e-5)
        {
          pos_r.Fill( 1.0/float(numberOfClasses) );
          qr.Fill( 1.0/float(numberOfClasses) );
          for(unsigned int c=0; c<numberOfClasses; c++)
            pos_change += std::pow(pos_r[c]-pos_ant[c],2.0);
        }
        else
        {
          for(unsigned int c=0; c<numberOfClasses; c++)
          {
            pos_r[c] = pos_r[c]/pr;
            qr[c] = pos_r[c];
            qr[c] = std::pow(qr[c],beta);
            //For convergence:
            pos_change += std::pow(pos_r[c]-pos_ant[c],2.0);
          }
        }
        pr = std::pow(pr,alpha);
        cost_func += pr;
        pos.Set(pos_r);
        q.Set(qr);
        r++;
      }
      ++q;
      ++b;
      ++f;
      ++pos;
      ++Omega;
    }
    std::cout << CurrentNumberOfIterations << "\t" << -std::log(cost_func) << "\t" << pos_change/float(OmegaSize) << std::endl;
    if (pos_change/float(OmegaSize)<1e-4)
      break;
  }

  //
  // Compute posteriors
  //
    typedef unsigned short  LabelType;
    typedef itk::BayesianClassifierImageFilter< VectorImageType,LabelType,
            float,float >   ClassifierFilterType;
    ClassifierFilterType::Pointer classifier = ClassifierFilterType::New();
    classifier->SetInput( bayesianInitializer->GetOutput() );
    classifier->SetPriors( priorImage );
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
