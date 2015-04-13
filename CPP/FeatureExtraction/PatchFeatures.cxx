// image storage and I/O classes
#include "itkSize.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"

#include "itkNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"

int main(int argc, char *argv[])
{
  const unsigned int Dimension = 3;
  typedef float ImagePixelType;
  typedef itk::Vector< ImagePixelType, 1 > MeasurementVectorType;
  typedef itk::Image< ImagePixelType, Dimension > ImageType;
  typedef itk::Image< MeasurementVectorType, Dimension > ArrayImageType;
  typedef itk::VectorImage< ImagePixelType, Dimension > VectorImageType;

  float alpha = 1;
  bool use_mask = false;

  if( argc < 3 )
    {
    std::cerr << "Usage: " << argv[0] << " InputImage OutputVectorImage [alpha_radius=1] [InputImageMask]" << std::endl;
    return EXIT_FAILURE;
    }

  if(argc>3)
  {
    alpha=atoi(argv[3]);
  }

  if(argc>4)
  {
    use_mask = true;
  }

  //
  // Load images
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
  // Output Vector Image
  //
  unsigned int d = static_cast<unsigned int>(pow(2*alpha+1,float(Dimension)));
  VectorImageType::PixelType tmp(d);
  tmp.Fill(0);
  VectorImageType::Pointer features = VectorImageType::New();
  features->SetOrigin( queryReader->GetOutput()->GetOrigin() );
  features->SetDirection( queryReader->GetOutput()->GetDirection() );
  features->SetSpacing( queryReader->GetOutput()->GetSpacing() );
  features->SetRegions( queryReader->GetOutput()->GetLargestPossibleRegion() );
  features->SetVectorLength(d);
  features->Allocate();
  features->FillBuffer(tmp);

  //
  // Valid region
  //
  ImageType::SizeType regionSize;
  ImageType::IndexType regionIndex;
  ImageType::SizeType radius;
  for (unsigned int dim = 0; dim<Dimension; dim++)
  {
    regionSize[dim] = queryReader->GetOutput()->GetLargestPossibleRegion().GetSize()[dim] - 2*alpha;
    regionIndex[dim] = alpha;
    radius[dim] = alpha;
  }

  //
  // Region/ball iterator
  //
  itk::ConstNeighborhoodIterator<ImageType> x(radius, queryReader->GetOutput(),queryReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<VectorImageType> f(features,features->GetLargestPossibleRegion());
  x.GoToBegin();
  f.GoToBegin();
  while(!x.IsAtEnd())
  {
    // Set the current pixel to white
    unsigned int allInBounds = 0;
    for(unsigned int rp = 0; rp < d; rp++)
    {
      bool IsInBounds;
      x.GetPixel(rp, IsInBounds);
      if(IsInBounds)
      {
        allInBounds++;
        tmp[rp] = x.GetPixel(rp);
      }
    }
    if( allInBounds==d )
    {
      f.Set( tmp );
    }
    ++x;
    ++f;
  }

  //
  //Write output
  //
  typedef itk::ImageFileWriter< VectorImageType >    WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[2] );
  writer->SetInput( features );
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
