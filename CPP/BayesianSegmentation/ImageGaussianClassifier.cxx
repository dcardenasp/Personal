/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// Software Guide : BeginLatex
//
// This example illustrates the use of the \doxygen{RGBGibbsPriorFilter}.
// The filter outputs a binary segmentation that can be improved by the
// deformable model. It is the first part of our hybrid framework.
//
// First, we include the appropriate header file.
//
// Software Guide : EndLatex

#include <iostream>
#include <string>
#include <math.h>

//Registration classes
#include "itkImageRegistrationMethodv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkVersorRigid3DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkResampleImageFilter.h"
#include "itkComposeImageFilter.h"

// classes help the Gibbs filter to segment the image
#include "itkImageClassifierBase.h"
#include "itkImageGaussianModelEstimator.h"
#include "itkGaussianMembershipFunction.h"
#include "itkMahalanobisDistanceMembershipFunction.h"
#include "itkMinimumDecisionRule.h"

// image storage and I/O classes
#include "itkSize.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#define   NUM_CLASSES         5
#define   MAX_NUM_ITER        1000

int main( int argc, char *argv[] )
{
  if( argc != 4 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " inputImage templateImage priorImage outputImage" << std::endl;
    return 1;
    }

  const unsigned short NUMBANDS = 1;
  const unsigned short NDIMENSION = 3;
  typedef float InputPixelType;

  typedef itk::Image< InputPixelType, NDIMENSION > ImageType;
  typedef itk::Image< unsigned short, NDIMENSION > ClassImageType;
  typedef itk::Image<itk::Vector<InputPixelType,NUMBANDS>,
          NDIMENSION> VecImageType;

  //
  // Load query, template and prior images.
  //
  typedef itk::ImageFileReader< ImageType >         ReaderType;
  typedef itk::ImageFileReader< VecImageType >      VectorReaderType;
  typedef itk::ImageFileWriter<  ClassImageType  >  WriterType;
  ReaderType::Pointer fixedImageReader = ReaderType::New();
  ReaderType::Pointer movingImageReader = ReaderType::New();
  VectorReaderType::Pointer priorImageReader = VectorReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  fixedImageReader->SetFileName( argv[1] );
  movingImageReader->SetFileName( argv[2] );
  priorImageReader->SetFileName( argv[3] );
  writer->SetFileName( argv[4] );
  try
  {
    fixedImageReader->Update();
    movingImageReader->Update();
    priorImageReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception thrown " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Images read." << std::endl;

  //
  // Template/Prior - Query Rigid Registration
  //
  std::cout << "Starting rigid registration" << std::endl;
  typedef itk::VersorRigid3DTransform< double > RigidTransformType;
  RigidTransformType::Pointer  initialTransform = RigidTransformType::New();
  typedef itk::CenteredTransformInitializer
          < RigidTransformType, ImageType, ImageType > TransformInitializerType;
  TransformInitializerType::Pointer initializer
          = TransformInitializerType::New();
  initializer->SetTransform(   initialTransform );
  initializer->SetFixedImage(  fixedImageReader->GetOutput() );
  initializer->SetMovingImage( movingImageReader->GetOutput() );
  initializer->MomentsOn();
  initializer->InitializeTransform();

  typedef itk::RegularStepGradientDescentOptimizerv4<double>    GDOptimizerType;
  GDOptimizerType::Pointer gdOptimizer = GDOptimizerType::New();
  typedef GDOptimizerType::ScalesType OptimizerScalesType;
  OptimizerScalesType optimizerScales(
          initialTransform->GetNumberOfParameters() );
  const double translationScale = 1.0 / 1000.0;
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;
  optimizerScales[5] = translationScale;
  gdOptimizer->SetScales( optimizerScales );
  gdOptimizer->SetNumberOfIterations( 1 );
  gdOptimizer->SetLearningRate( 0.2 );
  gdOptimizer->SetMinimumStepLength( 0.00001 );
  gdOptimizer->SetReturnBestParametersAndValue(true);

  typedef itk::MattesMutualInformationImageToImageMetricv4
          < ImageType, ImageType >                      MetricType;
  MetricType::Pointer metric = MetricType::New();

  typedef itk::ImageRegistrationMethodv4
          < ImageType, ImageType, RigidTransformType >  RigidRegistrationType;
  const unsigned int numberOfLevels = 1;
  RigidRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize( 1 );
  shrinkFactorsPerLevel[0] = 1;
  RigidRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize( 1 );
  smoothingSigmasPerLevel[0] = 0;
  RigidRegistrationType::Pointer rigidRegistration
          = RigidRegistrationType::New();
  rigidRegistration->SetMetric(             metric                        );
  rigidRegistration->SetOptimizer(          gdOptimizer                   );
  rigidRegistration->SetFixedImage(    fixedImageReader->GetOutput()      );
  rigidRegistration->SetMovingImage(     movingImageReader->GetOutput()   );
  rigidRegistration->SetInitialTransform(   initialTransform              );
  rigidRegistration->SetNumberOfLevels(     numberOfLevels                );
  rigidRegistration->SetSmoothingSigmasPerLevel(  smoothingSigmasPerLevel );
  rigidRegistration->SetShrinkFactorsPerLevel(    shrinkFactorsPerLevel   );
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
  const unsigned int numberOfClasses
              = priorImageReader->GetOutput()->GetNumberOfComponentsPerPixel();

  //
  // Prior rigid mapping
  //
  std::cout << "Starting the prior mapping" << std::endl;
  const RigidTransformType::ParametersType finalRigidParameters =
          rigidRegistration->GetOutput()->Get()->GetParameters();
  RigidTransformType::Pointer finalRigidTransform = RigidTransformType::New();
  finalRigidTransform->SetFixedParameters(
          rigidRegistration->GetOutput()->Get()->GetFixedParameters() );
  finalRigidTransform->SetParameters( finalRigidParameters );
  VecImageType::PixelType defaultPrior(numberOfClasses);
  defaultPrior.Fill(0.0); defaultPrior[0] = 1.0;
  typedef itk::ResampleImageFilter< VecImageType, VecImageType >
          VectorResampleFilterType;
  VectorResampleFilterType::Pointer vectorResampleFilter
          = VectorResampleFilterType::New();
  vectorResampleFilter->SetTransform( finalRigidTransform );
  vectorResampleFilter->SetInput( priorImageReader->GetOutput() );
  vectorResampleFilter->SetSize(
          fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize() );
  vectorResampleFilter->SetOutputOrigin(
          fixedImageReader->GetOutput()->GetOrigin() );
  vectorResampleFilter->SetOutputSpacing(
          fixedImageReader->GetOutput()->GetSpacing() );
  vectorResampleFilter->SetOutputDirection(
          fixedImageReader->GetOutput()->GetDirection() );
  vectorResampleFilter->SetDefaultPixelValue( defaultPrior );
  try
  {
    vectorResampleFilter->Update();
    std::cout << "Priors mapped" << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }
  ClassImageType::Pointer class0 = ClassImageType::New();

  //
  // Casting image to vector image
  //
  typedef itk::ComposeImageFilter< ImageType, VecImageType > ImageToVectorType;
  ImageToVectorType::Pointer img2vecCaster = ImageToVectorType::New();
  img2vecCaster->SetInput( fixedImageReader->GetOutput() );
  img2vecCaster->Update();
  VecImageType::Pointer vecImage = img2vecCaster->GetOutput();

  //
  //Set membership function (Using the statistics objects)
  //
  namespace stat = itk::Statistics;
  typedef stat::GaussianMembershipFunction< VecImageType::PixelType >
          MembershipFunctionType;
  typedef MembershipFunctionType::Pointer MembershipFunctionPointer;
  typedef std::vector< MembershipFunctionPointer >
    MembershipFunctionPointerVector;

  //
  // Set the image model estimator (train the class models)
  //
  typedef itk::ImageGaussianModelEstimator<VecImageType, MembershipFunctionType,
          ClassImageType> ImageGaussianModelEstimatorType;
  ImageGaussianModelEstimatorType::Pointer applyEstimateModel
          = ImageGaussianModelEstimatorType::New();
  applyEstimateModel->SetNumberOfModels( numberOfClasses );
  applyEstimateModel->SetInputImage( vecImage );
  applyEstimateModel->SetTrainingImage( vectorResampleFilter->GetOutput() );
  applyEstimateModel->Update();

  std::cout << " site 1 " << std::endl;

  applyEstimateModel->Print(std::cout);

  MembershipFunctionPointerVector membershipFunctions =
    applyEstimateModel->GetMembershipFunctions();

  std::cout << " site 2 " << std::endl;

  //----------------------------------------------------------------------
  //Set the decision rule
  //----------------------------------------------------------------------
  typedef itk::Statistics::DecisionRule::Pointer DecisionRuleBasePointer;

  typedef itk::Statistics::MinimumDecisionRule DecisionRuleType;
  DecisionRuleType::Pointer  myDecisionRule = DecisionRuleType::New();

  std::cout << " site 3 " << std::endl;

  //----------------------------------------------------------------------
  // Set the classifier to be used and assigne the parameters for the
  // supervised classifier algorithm except the input image which is
  // grabbed from the Gibbs application pipeline.
  //----------------------------------------------------------------------
  //---------------------------------------------------------------------

  //  Software Guide : BeginLatex
  //
  //  Then we define the classifier that is needed
  //  for the Gibbs prior model to make correct segmenting decisions.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::ImageClassifierBase< VecImageType,
                                    ClassImageType > ClassifierType;
  typedef ClassifierType::Pointer                    ClassifierPointer;
  ClassifierPointer myClassifier = ClassifierType::New();
  // Software Guide : EndCodeSnippet

  // Set the Classifier parameters
  myClassifier->SetNumberOfClasses(NUM_CLASSES);

  // Set the decison rule
  myClassifier->SetDecisionRule((DecisionRuleBasePointer) myDecisionRule );

  //Add the membership functions
  for( unsigned int i=0; i<NUM_CLASSES; i++ )
    {
    myClassifier->AddMembershipFunction( membershipFunctions[i] );
    }

  //Set the Gibbs Prior labeller
  //  Software Guide : BeginLatex
  //
  //  After that we can define the multi-channel Gibbs prior model.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::RGBGibbsPriorFilter<VecImageType,ClassImageType>
    GibbsPriorFilterType;
  GibbsPriorFilterType::Pointer applyGibbsImageFilter =
    GibbsPriorFilterType::New();
  // Software Guide : EndCodeSnippet

  // Set the MRF labeller parameters
  //  Software Guide : BeginLatex
  //
  //  The parameters for the Gibbs prior filter are defined
  //  below. \code{NumberOfClasses} indicates how many different objects are in
  //  the image.  The maximum number of iterations is the number of
  //  minimization steps.  \code{ClusterSize} sets the lower limit on the
  //  object's size.  The boundary gradient is the estimate of the variance
  //  between objects and background at the boundary region.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  applyGibbsImageFilter->SetNumberOfClasses(NUM_CLASSES);
  applyGibbsImageFilter->SetMaximumNumberOfIterations(MAX_NUM_ITER);
  applyGibbsImageFilter->SetClusterSize(10);
  applyGibbsImageFilter->SetBoundaryGradient(6);
  applyGibbsImageFilter->SetObjectLabel(1);
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  We now set the input classifier for the Gibbs prior filter and the
  //  input to the classifier. The classifier will calculate the mean and
  //  variance of the object using the class image, and the results will be
  //  used as parameters for the Gibbs prior model.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  applyGibbsImageFilter->SetInput(vecImage);
  applyGibbsImageFilter->SetClassifier( myClassifier );
  applyGibbsImageFilter->SetTrainingImage(trainingimagereader->GetOutput());
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  Finally we execute the Gibbs prior filter using the Update() method.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  applyGibbsImageFilter->Update();
  // Software Guide : EndCodeSnippet

  std::cout << "applyGibbsImageFilter: " << applyGibbsImageFilter;

  writer->SetInput( applyGibbsImageFilter->GetOutput() );
  writer->Update();

  //  Software Guide : BeginLatex
  //
  //  We execute this program on the image \code{brainweb89.png}. The
  //  following parameters are passed to the command line:
  //
  //  \small
  //  \begin{verbatim}
  //GibbsGuide.exe brainweb89.png brainweb89_train.png brainweb_gp.png
  //  \end{verbatim}
  //  \normalsize
  //
  //  \code{brainweb89train} is a training image that helps to estimate the object statistics.
  //
  //  Note that in order to successfully segment other images, one has to
  //  create suitable training images for them. We can also segment color
  //  (RGB) and other multi-channel images.
  //
  //  Software Guide : EndLatex

  return 0;
}
