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
#ifndef __itkParzenMembershipFunction_hxx
#define __itkParzenMembershipFunction_hxx

#include "itkParzenMembershipFunction.h"

namespace itk
{
namespace Statistics
{
template< typename TMeasurementVector >
ParzenMembershipFunction< TMeasurementVector >
::ParzenMembershipFunction()
{  
  MeasurementVectorType sample;
  sample.Fill( 0.0 );
  m_SampleList = SampleListType::New();
  m_SampleList->SetMeasurementVectorSize(this->GetMeasurementVectorSize());
  m_SampleList->Resize(1);
  m_SampleList->SetMeasurementVector(0,sample);

  m_PreFactor = 1.0 / std::sqrt(2.0 * vnl_math::pi); // default univariate

  m_Covariance.SetSize(this->GetMeasurementVectorSize(), this->GetMeasurementVectorSize());
  m_Covariance.SetIdentity();

  m_InverseCovariance = m_Covariance;

  m_CovarianceNonsingular = true;
}

template< typename TMeasurementVector >
void
ParzenMembershipFunction< TMeasurementVector >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  //os << indent << "Mean: " << m_Mean << std::endl;
  os << indent << "Samples: " << std::endl;
  os << indent << m_SampleList << std::endl;
  os << indent << "Covariance: " << std::endl;
  os << m_Covariance.GetVnlMatrix();
  os << indent << "InverseCovariance: " << std::endl;
  os << indent << m_InverseCovariance.GetVnlMatrix();
  os << indent << "Prefactor: " << m_PreFactor << std::endl;
  os << indent << "Covariance nonsingular: " <<
    (m_CovarianceNonsingular ? "true" : "false") << std::endl;
}

template< typename TMeasurementVector >
void
ParzenMembershipFunction< TMeasurementVector >
::SetSampleList( typename SampleListType::Pointer list)
{
  this->SetMeasurementVectorSize( list->GetMeasurementVectorSize() );
  m_SampleList->SetMeasurementVectorSize(this->GetMeasurementVectorSize());
  m_SampleList->Resize( list->Size() );
  for (unsigned int i=0; i<list->Size(); i++ )
  {
    m_SampleList->SetMeasurementVector(i, list->GetMeasurementVector(i) );

  }
  this->Modified();
}

template< typename TMeasurementVector >
void
ParzenMembershipFunction< TMeasurementVector >
::SetCovariance(const CovarianceMatrixType & cov)
{
  // Sanity check
  if ( cov.GetVnlMatrix().rows() != cov.GetVnlMatrix().cols() )
    {
    itkExceptionMacro(<< "Covariance matrix must be square");
    }
  if ( this->GetMeasurementVectorSize() )
    {
    if ( cov.GetVnlMatrix().rows() != this->GetMeasurementVectorSize() )
      {
      itkExceptionMacro(<< "Length of measurement vectors must be"
                        << " the same as the size of the covariance.");
      }
    }
  else
    {
    // not already set, cache the size
    this->SetMeasurementVectorSize( cov.GetVnlMatrix().rows() );
    }

  if (m_Covariance == cov)
    {
    // no need to copy the matrix, compute the inverse, or the normalization
    return;
    }

  m_Covariance = cov;

  // the inverse of the covariance matrix is first computed by SVD
  vnl_matrix_inverse< double > inv_cov( m_Covariance.GetVnlMatrix() );

  // the determinant is then costless this way
  double det = inv_cov.determinant_magnitude();

  if( det < 0.)
    {
    itkExceptionMacro( << "det( m_Covariance ) < 0" );
    }

  // 1e-6 is an arbitrary value!!!
  const double singularThreshold = 1.0e-6;
  m_CovarianceNonsingular = ( det > singularThreshold );

  if( m_CovarianceNonsingular )
    {
    // allocate the memory for m_InverseCovariance matrix
    m_InverseCovariance.GetVnlMatrix() = inv_cov.inverse();

    // calculate coefficient C of multivariate gaussian
    m_PreFactor =
      1.0 / ( std::sqrt(det) *
        std::pow( std::sqrt(2.0 * vnl_math::pi),
               static_cast< double >( this->GetMeasurementVectorSize() ) ) );
    }
  else
    {
    const double aLargeDouble = std::pow(NumericTraits<double>::max(), 1.0/3.0)
      / (double) this->GetMeasurementVectorSize();
    m_InverseCovariance.SetIdentity();
    m_InverseCovariance *= aLargeDouble;

    m_PreFactor = 1.0;
    }

  this->Modified();
}

template< typename TMeasurementVector >
inline double
ParzenMembershipFunction< TMeasurementVector >
::Evaluate(const MeasurementVectorType & measurement) const
{
  const MeasurementVectorSizeType measurementVectorSize =
    this->GetMeasurementVectorSize();

  // temp = ( y - mean )^t * InverseCovariance * ( y - mean )
  //
  // This is manually done to remove dynamic memory allocation:
  // double temp = dot_product( tempVector,  m_InverseCovariance.GetVnlMatrix() * tempVector );
  //  
  double summation = 0.0, temp;
  MeasurementVectorType sample;
  for(unsigned int i = 0; i<this->m_SampleList->Size(); i++)
  {
    sample = this->m_SampleList->GetMeasurementVector(i);
    temp = 0.0;
    for(MeasurementVectorSizeType r = 0; r < measurementVectorSize; ++r)
    {
      double rowdot = 0.0;
      for (MeasurementVectorSizeType c = 0; c < measurementVectorSize; ++c)
      {
        rowdot += m_InverseCovariance(r, c) * ( measurement[c] - sample[c] );
      }
      temp += rowdot * ( measurement[r] - sample[r] );
    }    
    summation += std::exp(-0.5 * temp);
  }

  return m_PreFactor * summation / static_cast<double>(this->m_SampleList->Size());
}

template< typename TVector >
typename LightObject::Pointer
ParzenMembershipFunction< TVector >
::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  typename Self::Pointer membershipFunction =
    dynamic_cast<Self *>(loPtr.GetPointer());
  if(membershipFunction.IsNull())
    {
    itkExceptionMacro(<< "downcast to type "
                      << this->GetNameOfClass()
                      << " failed.");
    }

  membershipFunction->SetMeasurementVectorSize( this->GetMeasurementVectorSize() );
  membershipFunction->SetSampleList( this->m_SampleList );
  membershipFunction->SetCovariance( this->GetCovariance() );

  return loPtr;
}
} // end namespace Statistics
} // end of namespace itk

#endif
