#include "itkVectorImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"
#include "itkNeighborhoodIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRandomIteratorWithIndex.h"

#include "itkCenteredTransformInitializer.h"
#include "itkVersorRigid3DTransform.h"
#include "itkAffineTransform.h"
#include "itkResampleImageFilter.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkImageRegistrationMethod.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"

#include "itkImageDuplicator.h"
#include "itkVectorImageToImageAdaptor.h"
#include "itkStatisticsImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkBayesianClassifierImageFilter.h"

#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <algorithm>
#include <vector>
#define PI 3.14159265

//#include "itkBayesianClassifierImageFilter.h"

//Gaussian Kernel
float kGauss(float x, float mu, float sigma){
  return pow((2*PI*pow(sigma,2.0)),-0.5)*exp(-pow(x-mu,2.0)/(2*pow(sigma,2.0)));
}


int main(int argc, char *argv[])
{
  if (argc < 2)
    {
    std::cerr << "Usage: " << argv[0] << " imageFilename priorList outputFilename method" << std::endl;
    return EXIT_FAILURE;
    }
  const char * inputImageFileName         = argv[1];
  const char * inputMovingImageFileNames  = argv[2];
  const char * outputImageFileName        = argv[3];
  int          method                     = atoi(argv[4]);
  //method = 0: 1-Normal by Class
  //method = 1: GMM by Class
  //method = 2: Parzen estimation by Class
  
  typedef itk::Image<float, 3>  ScalarImageType;
  typedef itk::Image<unsigned char, 3>  LabelImageType;

  typedef itk::ImageFileReader<ScalarImageType> ReaderType;
  ReaderType::Pointer fixed = ReaderType::New();
  fixed->SetFileName(inputImageFileName);
  fixed->Update();

  std::cout << "Input read\n";

  //*******LOAD AND CREATE PRIOR VECTOR IMAGE *********//
  std::vector<ScalarImageType::Pointer> prImages;
  std::ifstream infile(inputMovingImageFileNames);
  std::string line;
  int numClasses = 0;
  std::getline(infile, line);
  ReaderType::Pointer moving = ReaderType::New();
  moving->SetFileName(line);
  moving->Update();

  while (std::getline(infile, line))
  {
      ReaderType::Pointer tmp = ReaderType::New();
      tmp->SetFileName(line);
      tmp->Update();

      prImages.push_back(tmp->GetOutput());
      numClasses++;
  }

  float G[10][10]; //up to 10 classes. WARNING. Should be improved.
  for(int c1=0; c1<numClasses; c1++)
  for(int c2=0; c2<numClasses; c2++)
  {
    if(c1==c2)
      G[c1][c2] = 0.0;
    else if(abs(c1-c2)==1)
      G[c1][c2] = 0.5;
    else
      G[c1][c2] = 3.0;
  }

  std::cout << "Prior read and built\n";

  //******AFFINE REGISTERING PRIORS**********//
  typedef itk::VersorRigid3DTransform< double > RigidTransformType;
  typedef itk::AffineTransform< double, 3 > AffineTransformType;
  typedef itk::CenteredTransformInitializer< RigidTransformType,
          ScalarImageType, ScalarImageType > TransformInitializerType;
  typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;
  typedef itk::MattesMutualInformationImageToImageMetric<
          ScalarImageType, ScalarImageType > MetricType;

  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  RigidTransformType::Pointer  rigidTransform = RigidTransformType::New();
  initializer->SetTransform(   rigidTransform );
  initializer->SetFixedImage(  fixed->GetOutput() );
  initializer->SetMovingImage( moving->GetOutput() );
  initializer->MomentsOn();
  initializer->InitializeTransform();
  std::cout << "Rigid Transform Initialization completed" << std::endl;
  typedef itk:: LinearInterpolateImageFunction<
          ScalarImageType, double > InterpolatorType;
  typedef itk::ImageRegistrationMethod<
          ScalarImageType, ScalarImageType >    RegistrationType;
  MetricType::Pointer               metric        = MetricType::New();
  OptimizerType::Pointer            optimizer     = OptimizerType::New();
  InterpolatorType::Pointer         interpolator  = InterpolatorType::New();
  RegistrationType::Pointer         registration  = RegistrationType::New();
  registration->SetMetric(          metric        );
  registration->SetOptimizer(       optimizer     );
  registration->SetInterpolator(    interpolator  );
  registration->SetFixedImage(      fixed->GetOutput()  );
  registration->SetMovingImage(     moving->GetOutput() );
  metric->SetNumberOfHistogramBins( 50 );
  ScalarImageType::RegionType fixedRegion = fixed->GetOutput()->GetBufferedRegion();
  const unsigned int numberOfPixels = fixedRegion.GetNumberOfPixels();
  metric->ReinitializeSeed( 76926294 );
  registration->SetFixedImageRegion( fixedRegion );
  registration->SetInitialTransformParameters( rigidTransform->GetParameters() );
  registration->SetTransform( rigidTransform );
  typedef OptimizerType::ScalesType       OptimizerScalesType;
  OptimizerScalesType optimizerScales( rigidTransform->GetNumberOfParameters() );
  const double translationScale = 1.0 / 1000.0;
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;
  optimizerScales[5] = translationScale;
  optimizer->SetScales( optimizerScales );
  optimizer->SetMaximumStepLength( 0.2000  );
  optimizer->SetMinimumStepLength( 0.0001 );
  optimizer->SetNumberOfIterations( 500 );
  metric->SetNumberOfSpatialSamples( 10000L );
  registration->Update();
  rigidTransform->SetParameters( registration->GetLastTransformParameters() );
  std::cout << "Rigid Transform Completed: "
          << registration->GetOptimizer()->GetStopConditionDescription()
          << std::endl;

  AffineTransformType::Pointer  affineTransform = AffineTransformType::New();
  affineTransform->SetCenter( rigidTransform->GetCenter() );
  affineTransform->SetTranslation( rigidTransform->GetTranslation() );
  affineTransform->SetMatrix( rigidTransform->GetMatrix() );
  registration->SetTransform( affineTransform );
  registration->SetInitialTransformParameters( affineTransform->GetParameters() );
  optimizerScales = OptimizerScalesType( affineTransform->GetNumberOfParameters() );
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = 1.0;
  optimizerScales[4] = 1.0;
  optimizerScales[5] = 1.0;
  optimizerScales[6] = 1.0;
  optimizerScales[7] = 1.0;
  optimizerScales[8] = 1.0;
  optimizerScales[9]  = translationScale;
  optimizerScales[10] = translationScale;
  optimizerScales[11] = translationScale;
  optimizer->SetScales( optimizerScales );
  optimizer->SetMaximumStepLength( 0.2000  );
  optimizer->SetMinimumStepLength( 0.0001 );
  optimizer->SetNumberOfIterations( 500 );
  metric->SetNumberOfSpatialSamples( 50000L );
  registration->Update();
  affineTransform->SetParameters( registration->GetLastTransformParameters() );
  std::cout << "Affine Registration completed: "
            << registration->GetOptimizer()->GetStopConditionDescription()
            << std::endl;
  typedef itk::ComposeImageFilter<ScalarImageType> ImageToVectorImageFilterType;
  ImageToVectorImageFilterType::Pointer vectorBuilder = ImageToVectorImageFilterType::New();
  typedef itk::ResampleImageFilter< ScalarImageType,
          ScalarImageType>    ResampleFilterType;
  ResampleFilterType::Pointer resample = ResampleFilterType::New();
  resample->SetTransform( affineTransform );
  resample->SetSize(    fixed->GetOutput()->GetLargestPossibleRegion().GetSize() );
  resample->SetOutputOrigin(  fixed->GetOutput()->GetOrigin() );
  resample->SetOutputSpacing( fixed->GetOutput()->GetSpacing() );
  resample->SetOutputDirection( fixed->GetOutput()->GetDirection() );
  resample->SetInterpolator( interpolator );
  resample->SetDefaultPixelValue(1.0);
  for(unsigned int c=0; c<numClasses; c++)
  {
      resample->SetInput( prImages[c] );
      resample->Update();
      prImages[c] = resample->GetOutput();
      prImages[c]->DisconnectPipeline();      
      vectorBuilder->SetInput(c,prImages[c]);
      resample->SetDefaultPixelValue(0.0);
  }
  vectorBuilder->Update();
  typedef ImageToVectorImageFilterType::OutputImageType VectorImageType;
  VectorImageType::Pointer priors = vectorBuilder->GetOutput();
  typedef itk::ImageDuplicator< VectorImageType > DuplicatorType;
  DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(vectorBuilder->GetOutput());
  duplicator->Update();
  VectorImageType::Pointer posteriors = duplicator->GetOutput();

  //*******Partitioning the image********//
  itk::ImageRegionConstIterator<ScalarImageType> itFixed(fixed->GetOutput(),fixed->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<VectorImageType> itPr(priors,priors->GetLargestPossibleRegion());
  itk::ImageRegionIterator<VectorImageType> itPos(posteriors,posteriors->GetLargestPossibleRegion());
  VectorImageType::SizeType neighRadius; neighRadius.Fill(1);
  itk::NeighborhoodIterator<VectorImageType> itNeigh(neighRadius,posteriors,posteriors->GetLargestPossibleRegion());
  float s = 33.0, z=0;
  unsigned int i=0;
  VectorImageType::PixelType sumPr, h, means, cov, tmp, h_a, means_a, cov_a;
  sumPr.SetSize(numClasses);
  h.SetSize(numClasses);
  means.SetSize(numClasses);
  cov.SetSize(numClasses);
  tmp.SetSize(numClasses);
  sumPr.Fill(0.0);

  /*FOR PARZEN ESTIMATION:*/
  unsigned int M = 500;
  std::vector<ScalarImageType::IndexType> indexes(M);
  std::vector<VectorImageType::PixelType> w(M);
  std::vector<ScalarImageType::PixelType> f(M);

  if( method==2 )
  {
    itk::ImageRandomConstIteratorWithIndex<ScalarImageType> itRnd(fixed->GetOutput(),fixed->GetOutput()->GetLargestPossibleRegion());
    itRnd.SetNumberOfSamples(M);
    h.Fill(0.0);
    for(i=0,itRnd.GoToBegin(); !itRnd.IsAtEnd(); ++itRnd,i++)
    {
      indexes[i] = itRnd.GetIndex();
      itPos.SetIndex(indexes[i]);
      f[i] = itRnd.Get();
      w[i].SetSize(numClasses);
      w[i] = itPos.Get();
      h += w[i];
    }
    for(i=0; i<M; i++)
    {
      for(unsigned int c=0; c<numClasses; c++)
      w[i][c] = w[i][c]/h[c];
    }
  }
    
  /*for(itPr.GoToBegin(); !itPr.IsAtEnd(); ++itPr)
      sumPr = sumPr + itPr.Get();*/

  h.Fill(1.0/numClasses);
  means.Fill(0.0);
  cov.Fill(1000000.0);
  tmp.Fill(0.0);
  for(unsigned int iter = 0; iter<200; iter++)
  {

//E-step: Estimate q=Posterior
    for(itFixed.GoToBegin(), itPr.GoToBegin(), itPos.GoToBegin(); !itFixed.IsAtEnd(); ++itFixed, ++itPr, ++itPos)
    {
      z=0;
      switch(method)
      {
      case 0:
          for(unsigned int c=0; c<numClasses; c++)
          {
            tmp[c]=itPr.Get()[c]*h[c]*kGauss(itFixed.Get(), means[c], sqrt(cov[c]));
            z += tmp[c];
          }
          break;
      case 2:
          tmp.Fill(0.0);
          for (i=0;i<M;i++)
          {
            tmp = tmp + kGauss(itFixed.Get(),f[i],sqrt(cov[0]))*w[i];
          }
          for(unsigned int c=0; c<numClasses; c++)
          {
            tmp[c] = itPr.Get()[c]*h[c]*tmp[c];
            z += tmp[c];
          }
          break;
      }
      itPos.Set(tmp/z); //z: shouldn't sum up to zero
    }


    /*for(itNeigh.GoToBegin(); !itNeigh.IsAtEnd(); ++itNeigh)
    {
      tmp.Fill(0.0);
      VectorImageType::PixelType u;
      u.SetSize(numClasses);
      for(unsigned int r = 0; r < 27; r++)
      {
        bool IsInBounds;
        itNeigh.GetPixel(r, IsInBounds);
        if(IsInBounds)
        {
          tmp += itNeigh.GetPixel(r);
        }
        z = 0;
        for(unsigned int c1=0; c1<numClasses; c1++)
        {
          for(unsigned int c2=0; c2<numClasses; c2++)
          {
            u[c1] = exp(-G[c1][c2]*tmp[c2]);
          }
          u[c1] = u[c1]*itNeigh.GetCenterPixel()[c1];
          z += u[c1];
        }
      }
      itNeigh.SetCenterPixel(u/z);
    }*/

    h_a = h;
    means_a = means;
    cov_a = cov;

//M-step: Estimate parameters (theta)
    h.Fill(0.0);
    means.Fill(0.0);
    cov.Fill(0.0);
    tmp.Fill(0.0);

    for(itFixed.GoToBegin(), itPos.GoToBegin(); !itFixed.IsAtEnd(); ++itFixed, ++itPos)
    {
      h += itPos.Get();
      for(unsigned int c=0; c<numClasses; c++)
        means[c] += itPos.Get()[c]*itFixed.Get();
    }
    for(unsigned int c=0; c<numClasses; c++)
      means[c] = means[c]/h[c];

    for(itFixed.GoToBegin(), itPos.GoToBegin(); !itFixed.IsAtEnd(); ++itFixed, ++itPos)
    {
      for(unsigned int c=0; c<numClasses; c++)
        cov[c] += itPos.Get()[c]*pow(means[c]-itFixed.Get(),2.0);
    }
    for(unsigned int c=0; c<numClasses; c++)
      cov[c] = cov[c]/h[c];

    for(itPr.GoToBegin(); !itPr.IsAtEnd(); ++itPr)
    {
      z = 0;
      for(unsigned int c=0; c<numClasses; c++)
      {
        z += itPr.Get()[c]*h[c];
      }
      tmp = tmp + itPr.Get()/z;
    }
    z=0;
    for(unsigned int c=0; c<numClasses; c++)
    {
      h[c] = h[c]/tmp[c];
      z += h[c];
    }
    h = h/z;

//Convergence check:
    float h_d=0, h_n=0, means_d=0, means_n=0, cov_d=0, cov_n=0;
    for(unsigned int c=0; c<numClasses; c++)
    {
      h_d += pow(h[c]-h_a[c],2.0);
      h_n += pow(h[c],2.0);
      means_d += pow(means[c]-means_a[c],2.0);
      means_n += pow(means[c],2.0);
      cov_d += pow(cov[c]-cov_a[c],2.0);
      cov_n += pow(cov[c],2.0);
    }
    h_d = (h_d/h_n + means_d/means_n + cov_d/cov_n)/3;


    std::cout << "Iter " << iter << "\tConvergence " << h_d << std::endl
              << "Mixtures " << h << std::endl
              << "Means " << means << std::endl
              << "Cov " << cov << std::endl;

    if (h_d < 1e-4)
        break;


/*    for(itFixed.GoToBegin(), itPos.GoToBegin(); !itFixed.IsAtEnd(); ++itFixed, ++itPos)
    {
      r++;
      h = h + itPos.Get();
      VectorImageType::PixelType delta = itFixed.Get()*itPos.Get()-means;
      means = means + delta/r;
      for(unsigned int c=0; c<numClasses; c++)
        cov[c] = cov[c] + delta[c]*(itFixed.Get()*itPos.Get()[c]-means[c]);
    }
    cov = cov/(r-1);
    float sh=0;
    for(unsigned int c=0; c<numClasses; c++)
        sh += h[c];
    h = h/sh;*/


    /*for(itFixed.GoToBegin(), itPr.GoToBegin(), itPos.GoToBegin(); !itFixed.IsAtEnd(); ++itFixed, ++itPr, ++itPos)
    {
        float sumQ = 0;
        tmp.Fill(0.0);
        for (unsigned int c=0;c<numClasses;c++)
        {
          tmp[i]=pow((2*PI*cov[c]),-0.5)*exp(-pow(itFixed.Get()-means[c],2.0)/(2*cov[c]));
          if(itPr.Get()[c]>0)
            tmp[c] = tmp[c]*h[c]*(itPr.Get()[c])/sumPr[c];
          else
            tmp[c] = tmp[c]*h[c]*(1e-4)/sumPr[c];
          sumQ += tmp[c];
        }
        /*for (i=0;i<600;i++)
        {
            t[i]=pow((2*PI*pow(s,2.0)),-0.5)*exp(-pow(itFixed.Get()-f[i],2.0)/(2*pow(s,2.0)));
            tmp = tmp + t[i]*cPos[i];
        }
        itPos.Set(tmp/sumQ);
        std::cout << itPr.Get() << std::endl;
    }*/
  }

  std::cout << "Partitioning completed" << std::endl;

  //*/
  //*******Labeling the image********//
  LabelImageType::Pointer lbl = LabelImageType::New();
  lbl->SetRegions(   fixed->GetOutput()->GetLargestPossibleRegion() );
  lbl->SetOrigin(    fixed->GetOutput()->GetOrigin() );
  lbl->SetSpacing(   fixed->GetOutput()->GetSpacing() );
  lbl->SetDirection( fixed->GetOutput()->GetDirection() );
  lbl->Allocate();
  itk::ImageRegionIterator<LabelImageType> itLbl(lbl,lbl->GetLargestPossibleRegion());
  for(itLbl.GoToBegin(), itPos.GoToBegin(); !itPos.IsAtEnd(); ++itLbl, ++itPos)
  {
      unsigned char etiq = 0;
      for(unsigned int c=1; c<numClasses; c++)
      {
          etiq = (itPos.Get()[c]>itPos.Get()[etiq])?c:etiq;
      }
      itLbl.Set(etiq);      
  }
  //***************WRITER**************//

  typedef itk::ImageFileWriter< LabelImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(lbl);
  writer->SetFileName(outputImageFileName);
  writer->Update();

  //std::cout << pr->GetNumberOfComponentsPerPixel() << std::endl;
  //std::cout << reader->GetOutput()->GetLargestPossibleRegion() << std::endl;
  
  return EXIT_SUCCESS;
}
