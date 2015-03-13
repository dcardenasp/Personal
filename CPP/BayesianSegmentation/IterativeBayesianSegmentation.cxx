//Basics
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
#include "itkImageRegistrationMethodv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkVersorRigid3DTransform.h"
#include "itkBSplineTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkResampleImageFilter.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"

#include "itkImageRegionConstIterator.h"

#include "itkBayesianClassifierInitializationImageFilter.h"
#include "itkBayesianClassifierImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkWeightedMeanSampleFilter.h"
#include "itkWeightedCovarianceSampleFilter.h"
#include "itkGaussianMembershipFunction.h"
#include "itkWeightedParzenMembershipFunction.h"
#include "itkSampleClassifierFilter.h"

//  The following section of code implements a Command observer
//  used to monitor the evolution of the registration process.
//
#include "itkCommand.h"
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
    std::cout << optimizer->GetCurrentPosition() << std::endl;
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

  if( argc < 5 )
    {
    std::cerr << "Usage: " << argv[0] << " InputImage TemplateImage PriorVectorImage OutputImage [memfunc] [alpha]" << std::endl;
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
  const unsigned int numberOfClasses = priorReader->GetOutput()->GetNumberOfComponentsPerPixel();
  std::cout << "Scales:  " << gdOptimizer->GetScales() << std::endl;

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
  // Computing mask for all upcoming estimations.
  //
  typedef unsigned short  LabelType;
  typedef itk::BayesianClassifierImageFilter< VectorImageType,LabelType,
          float,float >   ClassifierFilterType;
  ClassifierFilterType::Pointer classifier1 = ClassifierFilterType::New();
  classifier1->SetInput( vectorResampleFilter->GetOutput() );
  typedef ClassifierFilterType::OutputImageType      ClassifierOutputImageType;
  typedef itk::BinaryThresholdImageFilter <ClassifierOutputImageType, ClassifierOutputImageType>
      BinaryThresholdImageFilterType;
  BinaryThresholdImageFilterType::Pointer threshold1
          = BinaryThresholdImageFilterType::New();
  threshold1->SetInput(classifier1->GetOutput());
  threshold1->SetLowerThreshold(0);
  threshold1->SetUpperThreshold(0);
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
  typedef itk::Statistics::ImageToListSampleAdaptor< ArrayImageType > SampleType;
  SampleType::Pointer sample = SampleType::New();
  sample->SetImage( caster->GetOutput() );

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


  //
  // Main loop
  //
  typedef itk::BayesianClassifierInitializationImageFilter< ImageType >
          BayesianInitializerType;
  BayesianInitializerType::Pointer bayesianInitializer
          = BayesianInitializerType::New();
  unsigned int MaximumNumberOfIterations = 10;
  for(unsigned int CurrentNumberOfIterations=0;
      CurrentNumberOfIterations < MaximumNumberOfIterations;
      CurrentNumberOfIterations++)
  {
    std::cout << CurrentNumberOfIterations << "/" << MaximumNumberOfIterations << std::endl;
  //
  // Parameter estimation
  //
    typedef itk::Statistics::ImageToListSampleAdaptor< ArrayImageType > SampleType;
    typedef itk::Statistics::WeightedCovarianceSampleFilter< SampleType >
            WeightedCovarianceAlgorithmType;
    std::vector< WeightedCovarianceAlgorithmType::Pointer > parameterEstimators;
    typedef itk::Statistics::GaussianMembershipFunction< MeasurementVectorType >
            MembershipFunctionType;
    typedef BayesianInitializerType::MembershipFunctionContainerType
            MembershipFunctionContainerType;
    MembershipFunctionContainerType::Pointer membershipFunctionContainer
            = MembershipFunctionContainerType::New();
    membershipFunctionContainer->Reserve(numberOfClasses);
    itk::ImageRegionIterator<VectorImageType> q(hidden, hidden->GetLargestPossibleRegion());
    itk::ImageRegionIterator<VectorImageType> p(joint, joint->GetLargestPossibleRegion());
    itk::ImageRegionIterator<VectorImageType> pos(posterior, posterior->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<VectorImageType> b(vectorResampleFilter->GetOutput(), vectorResampleFilter->GetOutput()->GetLargestPossibleRegion());
    unsigned int numberOfSamples = 1000;

    switch(memFunc)
    {
    case 0:
      for ( unsigned int c = 0; c < numberOfClasses; ++c )
      {
        std::cout << "Class " << c << " :" << std::endl;
        MembershipFunctionType::Pointer membershipFunction =
                MembershipFunctionType::New();
        WeightedCovarianceAlgorithmType::WeightArrayType weights;
        weights.SetSize(sample->Size());
        q.GoToBegin();
        p.GoToBegin();
        b.GoToBegin();
        unsigned int r=0;
        while(!q.IsAtEnd())
        {
          VectorImageType::PixelType qr = q.Get();
          VectorImageType::PixelType pr = p.Get();
          VectorImageType::PixelType br = b.Get();
          if (CurrentNumberOfIterations==0)
            weights[r] = br[c];
          else
            weights[r] = pow(pr[c]/(qr[c]+zero),alpha);
          ++q;
          ++p;
          ++b;
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
        membershipFunction->SetMean( parameterEstimators[c]->GetMean() );
        membershipFunction->SetCovariance( parameterEstimators[c]->GetCovarianceMatrix() );
        membershipFunctionContainer->SetElement( c, membershipFunction.GetPointer() );
      }
      break;
    case 1:
      std::cout << "Selecting samples for building the pdf" << std::endl;
      itk::ImageRegionConstIteratorWithIndex<VectorImageType> itIndVec(vectorResampleFilter->GetOutput(), vectorResampleFilter->GetOutput()->GetLargestPossibleRegion());
      itk::ImageRandomNonRepeatingConstIteratorWithIndex<ArrayImageType>
              itRndImg(caster->GetOutput(), caster->GetOutput()->GetLargestPossibleRegion());
      itRndImg.SetNumberOfSamples( numberOfSamples );
      typedef itk::Statistics::WeightedParzenMembershipFunction< MeasurementVectorType >
              WParzenMembershipFunctionType;
      WParzenMembershipFunctionType::Pointer wpmf = WParzenMembershipFunctionType::New();
      std::vector<WParzenMembershipFunctionType::WeightArrayType> wpmfWeights(numberOfClasses);
      std::vector<ArrayImageType::PixelType> samplelist(numberOfSamples);
      WParzenMembershipFunctionType::CovarianceMatrixType wpmfCov = wpmf->GetCovariance();
      wpmfCov[0][0] = std::pow(40.0,2.0);
      for(unsigned int c=0; c<numberOfClasses; c++)
      {
        (wpmfWeights[c]).SetSize(numberOfSamples);
      }
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
      for (unsigned int c=0; c<numberOfClasses; c++)
      {
        wpmf->SetSampleList( samplelist );
        wpmf->SetWeights(wpmfWeights[c]);
        wpmf->SetCovariance(wpmfCov);
        membershipFunctionContainer->SetElement( c, wpmf.GetPointer() );
      }
      std::cout << "Samples selected." << std::endl;
      break;
    }

  //
  // Compute class likelihoods
  //
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
    itk::ImageRegionConstIterator<VectorImageType> f(bayesianInitializer->GetOutput(), bayesianInitializer->GetOutput()->GetLargestPossibleRegion());
    f.GoToBegin();
    b.GoToBegin();
    q.GoToBegin();
    pos.GoToBegin();
    unsigned int r=0;
    float exp = 1.0;
    if(alpha!=1.0)
      exp = (alpha-1)/alpha;

    while(!q.IsAtEnd())
    {
      VectorImageType::PixelType qr = q.Get();
      VectorImageType::PixelType br = b.Get();
      VectorImageType::PixelType fr = f.Get();
      VectorImageType::PixelType pos_r = pos.Get();
      float z=0;
      for(unsigned int c=0; c<numberOfClasses; c++)
      {
        pos_r[c] = br[c]*fr[c];
        z += pos_r[c];
      }
      if(z<1e-5)
      {
        pos_r[0] = 1.0;
        qr[0] = 1.0;
        for(unsigned int c=1; c<numberOfClasses; c++)
        {
          pos_r[c] = 0.0;
          qr[c] = 0.0;
        }
      }
      else
      {
        for(unsigned int c=0; c<numberOfClasses; c++)
        {
          pos_r[c] /= z;
          qr[c] = pow(pos_r[c],exp);
        }
      }
      pos.Set(pos_r);
      q.Set(qr);
      ++q;
      ++b;
      ++f;
      ++pos;
    }

  //
  // RE-Align posteriors
  //
  /*  std::cout << "Posterior Realign" << std::endl;
    const unsigned int SplineOrder = 3;
    typedef itk::VectorIndexSelectionCastImageFilter<VectorImageType, ImageType>
            ImageExtractComponentFilterType;
    typedef itk::BSplineTransform< double, Dimension, SplineOrder >
            DeformableTransformType;
    DeformableTransformType::Pointer  bsplineTransformCoarse = DeformableTransformType::New();
    typedef itk::ImageRegistrationMethodv4
            < ImageType, ImageType, DeformableTransformType >          DeformableRegistrationType;
    DeformableRegistrationType::Pointer defRegistration = DeformableRegistrationType::New();
    unsigned int numberOfGridNodesInOneDimensionCoarse = 5;
    DeformableTransformType::PhysicalDimensionsType   fixedPhysicalDimensions;
    DeformableTransformType::MeshSizeType             meshSize;
    DeformableTransformType::OriginType               fixedOrigin;
    for( unsigned int i=0; i< Dimension; i++ )
    {
      fixedOrigin[i] = queryReader->GetOutput()->GetOrigin()[i];
      fixedPhysicalDimensions[i] = queryReader->GetOutput()->GetSpacing()[i] *
          static_cast<double>(
          queryReader->GetOutput()->GetLargestPossibleRegion().GetSize()[i] - 1 );
    }
    meshSize.Fill( numberOfGridNodesInOneDimensionCoarse - SplineOrder );
    bsplineTransformCoarse->SetTransformDomainOrigin( fixedOrigin );
    bsplineTransformCoarse->SetTransformDomainPhysicalDimensions(
                fixedPhysicalDimensions );
    bsplineTransformCoarse->SetTransformDomainMeshSize( meshSize );
    bsplineTransformCoarse->SetTransformDomainDirection(
                queryReader->GetOutput()->GetDirection() );
    typedef DeformableTransformType::ParametersType     ParametersType;
    unsigned int numberOfBSplineParameters = bsplineTransformCoarse->GetNumberOfParameters();
    optimizerScales = OptimizerScalesType( numberOfBSplineParameters );
    optimizerScales.Fill( 1.0 );
    gdOptimizer->SetScales( optimizerScales );
    ParametersType initialDeformableTransformParameters( numberOfBSplineParameters );
    initialDeformableTransformParameters.Fill( 0.0 );
    bsplineTransformCoarse->SetParameters( initialDeformableTransformParameters );
    defRegistration->SetMetric(             metric                      );
    defRegistration->SetOptimizer(          gdOptimizer                 );
    defRegistration->SetFixedImage(         queryReader->GetOutput()    );
    defRegistration->SetInitialTransform(   bsplineTransformCoarse    );
    gdOptimizer->SetMinimumStepLength(  0.01 );
    gdOptimizer->SetNumberOfIterations( 200 );
    //metric->SetNumberOfSpatialSamples( numberOfBSplineParameters * 100 );

    ImageExtractComponentFilterType::Pointer adaptor = ImageExtractComponentFilterType::New();
    adaptor->SetInput(posterior);
    for(unsigned int c=0; c<numberOfClasses; c++)
    {
      std::cout << "Class " << c << ":" << std::endl;
      adaptor->SetIndex(c);
      defRegistration->SetMovingImage( adaptor->GetOutput() );
      defRegistration->Update();

      std::cout << rigidRegistration->GetOutput()->Get()->GetParameters() << std::endl;
    }*/
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
