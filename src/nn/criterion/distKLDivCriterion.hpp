

#include "criterion.hpp"


namespace nn{
  
  template<typename T>
  class DistKLDivCriterion : public nn::Criterion {
    
    float error;
    Tensor<T> output;
    Tensor<T> gradInput;
    
  public:
    
    DistKLDivCriterion() = default;
    T updateOutput(Tensor<T>& input,Tensor<T>& target);
    Tensor<T>& updateGradInput();
    
  };
  
  template<typename T>
  T DistKLDivCriterion<T>::updateOutput(Tensor<T>& input, Tensor<T>& target){
    
    /*
       loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))
    */
    
    output.resize({input.shape.dimensions[0],input.shape.dimensions[1]});
    
    Tensor<T> t(output.dimensions);
    input.copy(output);
    Tensor<T> output(input.shape.dimensions);
    
    output.log(2,t);
    t.sub(target);
    output.mul(t);
    error = output.mean();
    
    return error;
  };
  
  template<typename T> Tensor<T>& DistKLDivCriterion<T>::updateGradInput(){
    
    gradInput.resize(output.shape.dimensions);
    gradInput.fill(error / output.shape.elements);
    gradInput.print();
    
    return gradInput;
    
  };
  
  
};
