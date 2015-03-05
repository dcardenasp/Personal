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
#ifndef __itkWeightedParzenMembershipFunction_h
#define __itkWeightedParzenMembershipFunction_h

#include "itkMatrix.h"
#include "itkListSample.h"
#include "itkMembershipFunctionBase.h"

namespace itk
{
namespace Statistics
{

template< typename TMeasurementVector >
class WeightedParzenMembershipFunction:
  public MembershipFunctionBase< TMeasurementVector >
{
public:
  /** Standard class typedefs */
  typedef WeightedParzenMembershipFunction                   Self;
  typedef MembershipFunctionBase< TMeasurementVector > Superclass;
  typedef SmartPointer< Self >                         Pointer;
  typedef SmartPointer< const Self >                   ConstPointer;

  /** Standard macros */
  itkTypeMacro(WeightedParzenMembershipFunction, MembershipFunction);
  itkNewMacro(Self);

  /** SmartPointer class for superclass */
  typedef typename Superclass::Pointer MembershipFunctionPointer;

  /** Typedef alias for the measurement vectors */
  typedef TMeasurementVector MeasurementVectorType;

  /** Length of each measurement vector */
  typedef typename Superclass::MeasurementVectorSizeType MeasurementVectorSizeType;

  /** Type of the mean vector. RealType on a vector-type is the same
   * vector-type but with a real element type.  */    
  typedef ListSample< MeasurementVectorType  > SampleListType;

  /** Type of the covariance matrix */
  typedef VariableSizeMatrix< double > CovarianceMatrixType;

  /** Set the samples of the Gaussian distribution. Mean is a vector type
   * similar to the measurement type but with a real element type. */
  void SetSampleList( typename SampleListType::Pointer list );

  typedef Array< typename MeasurementVectorType::ComponentType > WeightArrayType;
  void SetWeights( WeightArrayType weights );


  /** Set the covariance matrix. Covariance matrix is a
   * VariableSizeMatrix of doubles. The inverse of the covariance
   * matrix and the normlization term for the multivariate Gaussian
   * are calculate whenever the covaraince matrix is changed. */
  void SetCovariance(const CovarianceMatrixType & cov);

  /* Get the covariance matrix. Covariance matrix is a
  VariableSizeMatrix of doubles. */
  itkGetConstReferenceMacro(Covariance, CovarianceMatrixType);

  /* Get the inverse covariance matrix. Covariance matrix is a
  VariableSizeMatrix of doubles. */
  itkGetConstReferenceMacro(InverseCovariance, CovarianceMatrixType);

  /** Evaluate the probability density of a measurement vector. */
  double Evaluate(const MeasurementVectorType & measurement) const;

  /** Method to clone a membership function, i.e. create a new instance of
   * the same type of membership function and configure its ivars to
   * match. */
  virtual typename LightObject::Pointer InternalClone() const;

protected:
  WeightedParzenMembershipFunction(void);
  virtual ~WeightedParzenMembershipFunction(void) {}
  void PrintSelf(std::ostream & os, Indent indent) const;

private:
  WeightedParzenMembershipFunction(const Self &);   //purposely not implemented
  void operator=(const Self &); //purposely not implemented

  typename SampleListType::Pointer m_SampleList;      // samples
  WeightArrayType                  m_Weights;         // weights
  CovarianceMatrixType             m_Covariance;      // covariance matrix

  // inverse covariance matrix. automatically calculated
  // when covariace matirx is set.
  CovarianceMatrixType m_InverseCovariance;

  // pre_factor (normalization term). automatically calculated
  // when covariace matrix is set.
  double m_PreFactor;

  /** Boolean to cache whether the covarinace is singular or nearly singular */
  bool m_CovarianceNonsingular;
};
} // end of namespace Statistics
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkWeightedParzenMembershipFunction.hxx"
#endif

#endif
