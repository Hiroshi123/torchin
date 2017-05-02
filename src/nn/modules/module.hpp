
#pragma once

//#include <iostream>
#include <vector>
#include <math.h>

#include "tensor.hpp"

namespace nn {
  
  class Module{
    
    friend class Sequential;
    friend class Parallel;
    friend class Concat;
    
    
  private:
    
    bool train = false;
    Tensor<> gradInput;
    Tensor<> output;
    
    virtual Tensor<>& updateOutput(Tensor<>&) = 0;
    virtual Tensor<>& updateGradInput(Tensor<>& gradin) = 0;
    virtual void accGradParameters(){};
    virtual void name() const{std::cout << "this should not be called :(( " << std::endl;};
    
    
  protected:
    
    
  public:
    
    Module();
    
    void parameters();
    
    //virtual Tensor<>* updateOutput(Tensor<>&) = 0;
    //virtual Tensor<>* updateOutput(Tensor<>&);
    //virtual void forward() = 0;
    
    template<typename T>
    Tensor<T>& forward(Tensor<T>& input);
    
    template<typename T>
    Tensor<T>& backward(Tensor<T>& gradInput);
    
    //virtual void backwardUpdate(Tensor<>&){};
    //virtual void accGradParameters(Tensor<>& input,Tensor<>& gradin){};
    
    //virtual void accGradParameters() = 0;
    //virtual void accUpdateGradParameters() = 0;
    
    //virtual void name() = 0;
    
    void training();
    void evaluate();
    
  };
  
  
  Module::Module()
  {
    //gradInput->print();
  }
  
  template<typename T>
  Tensor<T>& Module::forward(Tensor<T>& input){
    //input.print();
    return updateOutput(input);
  }
  
  template<typename T>
  Tensor<T>& Module::backward(Tensor<T>& gradInput){
    
    updateGradInput(gradInput);
    
    //accGradParameters();
    return gradInput;
    
  }
  
  void Module::parameters(){
    
  }
  
  void Module::training(){
    train = true;
  }
  
  void Module::evaluate(){
    train = false;
  }
  
};
