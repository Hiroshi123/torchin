
//#include "tensor.hpp"

#pragma once

namespace nn {
  
  class Criterion {
    
  public:
    
    //Tensor<> output;
    
    Criterion(){};
    
    float forward(Tensor<>& input,Tensor<>& target);
    Tensor<>& backward();
    
    virtual float updateOutput(Tensor<>&,Tensor<>&) = 0;
    virtual Tensor<>& updateGradInput() = 0;
    
    
  };
  
  
  // inline float Criterion::updateOutput(Tensor<>& input,Tensor<>& target){
    
  //   //updateOutput(Tensor<> gradInput);
    
  //   return 0.0;
  // }
  
  
  float Criterion::forward(Tensor<>& input,Tensor<>& target){
    
    return updateOutput(input,target);
    
  };
  
  Tensor<>& Criterion::backward(){
    
    return updateGradInput();
    
  };
  
};

