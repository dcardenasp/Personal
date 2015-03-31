#include "itkGaussianMembershipFunction.h"
#include "itkPointSet.h"

#include "itkParzenMembershipFunction.h"
#include "itkWeightedParzenMembershipFunction.h"

int main( int argc, char *argv[] )
{
    typedef double PixelType;
    const unsigned int ImageDimension = 3;
    typedef itk::Vector<PixelType, ImageDimension> VectorType;
    typedef itk::Statistics::ListSample<VectorType> SampleType;
    typedef itk::Statistics::GaussianMembershipFunction<VectorType>
            GaussianMembershipFunctionType;
    SampleType::Pointer sample = SampleType::New();
    GaussianMembershipFunctionType::Pointer gmf
            = GaussianMembershipFunctionType::New();
    GaussianMembershipFunctionType::CovarianceMatrixType cov = gmf->GetCovariance();

    sample->Resize(2);

    VectorType v1;
    v1[0] = 0.0;
    v1[1] = 0.0;
    v1[2] = 0.0;
    sample->SetMeasurementVector(0,v1);

    VectorType v2;
    v2[0] = 1.0;
    v2[1] = 4.0;
    v2[2] = 3.0;
    sample->SetMeasurementVector(1,v2);

    VectorType v3;
    v3[0] = -1.0;
    v3[1] = 3.0;
    v3[2] = 4.0;

    std::vector<VectorType> samplelist(2);
    samplelist[0] = v1;
    samplelist[1] = v2;

    typedef itk::Statistics::ParzenMembershipFunction< VectorType > PMFType;
    PMFType::Pointer pmf = PMFType::New();
    pmf->SetSampleList( sample );

    typedef itk::Statistics::WeightedParzenMembershipFunction< VectorType > WPMFType;
    WPMFType::Pointer wpmf = WPMFType::New();
    wpmf->SetSampleList( samplelist );
    WPMFType::WeightArrayType weights;
    weights.SetSize( 2 );
    weights.SetElement( 0, 1.0 );
    weights.SetElement( 1, 1.0 );
    wpmf->SetWeights( weights );

    for(int d=0; d<ImageDimension; d++)
        cov[d][d] *= atof(argv[1]);

    gmf->SetCovariance( cov );
    pmf->SetCovariance( cov );
    wpmf->SetCovariance( cov );

    double s=0;
    for (int i=0; i<2; i++)
    {
        gmf->SetMean(sample->GetMeasurementVector(i));
        s += gmf->Evaluate( v3 );
    }

    std::cout << s/2.0 << std::endl;
    std::cout << pmf->Evaluate(v3) << std::endl;
    std::cout << wpmf->Evaluate(v3) << std::endl;

    /*
    typedef itk::PointSet<PixelType, ImageDimension> PointSetType;
    typedef PointSetType::PointType PointType;
    PointSetType::Pointer  PointSet = PointSetType::New();

    typedef PointSetType::PointsContainerPointer PointsContainerPointer;
    PointsContainerPointer  points = PointSet->GetPoints();
    // Create points
    typedef PointSetType::PointType PointType;
    PointType p0, p1, p2;
    p0[0]=  0.0; p0[1]= 0.0; p0[2]= 0.0;
    p1[0]=  0.1; p1[1]= 0.0; p1[2]= 0.0;
    p2[0]=  0.0; p2[1]= 0.1; p2[2]= 0.0;
    points->InsertElement(0, p0);
    points->InsertElement(1, p1);
    points->InsertElement(2, p2);
    PointSet->SetPoints(points);


    typedef itk::ManifoldParzenWindowsPointSetFunction<PointSetType> ParzenFilterType;
    typename ParzenFilterType::Pointer parzen = ParzenFilterType::New();
    parzen->SetRegularizationSigma( 1 );
    parzen->SetKernelSigma( 1 );
    parzen->SetEvaluationKNeighborhood( 3 );
    parzen->SetCovarianceKNeighborhood( 3 );
    parzen->SetUseAnisotropicCovariances( true );
    parzen->SetInputPointSet( PointSet );
    std::cout << parzen->Evaluate( p1 ) << std::endl;*/

    return EXIT_SUCCESS;
}
