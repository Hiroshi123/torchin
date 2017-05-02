
#include "module.hpp"

namespace nn{
  
  template<typename T>
  class Linear : public nn::Module {
    
  private:
    Tensor<T> weight;
    Tensor<T> delta_weight;
    Tensor<T> bias;
    Tensor<T> delta_bias;
    Tensor<T> output;
    Tensor<T> gradInput;
    Tensor<T> buffer;
    const std::function<T(T)> mulWithLR = [](T x) -> T {return x * 0.1;};
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    //void accGradParameters(Tensor<T>& input,Tensor<T>& gradInput);
    void accGradParameters();
    
    void reset();
    void name() const;
    
  protected:
    
  public:
    
    Linear(const size_t,const size_t);
    
    
  };
  
  template<typename T>
  Linear<T>::Linear(const size_t o,const size_t i){
    
    weight.resize({i,o});
    delta_weight.resize({i,o});
    bias.resize({i,1});
    delta_bias.resize({i,1});
    reset();
    
  };
  
  template<typename T>
  void Linear<T>::reset() {
    
    weight.rand();
    bias.rand();
    //output.fill(0);
    
  };
  
  template<typename T>
  Tensor<T>& Linear<T>::updateOutput(Tensor<T>& input){
    
    //std::cout << output.shape.elements << std::endl;
    
    buffer.resize({input.shape.dimensions[0],weight.shape.dimensions[0]});
    
    output.resize(input.shape.dimensions);
    input.copy(output);
    
    input.dot(weight,buffer,"RR");
    
    //input.dot(weight,buffer);
    //assert(input.shape.dimensions == 1 || input.shape.dimensions == 2);
    //assert(input.shape.dimensions == 1 || input.shape.dimensions == 2);
    //input.mul(weight);
    
    return buffer;
    
  };
  
  
  template<typename T>
  Tensor<T>& Linear<T>::updateGradInput(Tensor<T>& input){
    
    gradInput.resize(input.shape.dimensions);
    input.copy(gradInput);
    
    buffer.resize({input.shape.dimensions[0],weight.shape.dimensions[1]});
    
    // Tensor<T> t_weight({weight.shape.dimensions[1],weight.shape.dimensions[0]});
    // weight.transpose(t_weight);
    // input.dot(t_weight,buffer);
    
    input.dot(weight,buffer,"RC");
    
    return buffer;
    
  };
  
  template<typename T>
  void Linear<T>::accGradParameters(){
    
    output.dot(gradInput,delta_weight,"CC");
    delta_weight.apply(mulWithLR);
    weight.sub(delta_weight);
    
  };
  
  template<typename T>
  void Linear<T>::name() const {
    std::cout << "Linear (" << weight.shape.dimensions[0] << "x"
	      << weight.shape.dimensions[1] << ")"  << std::endl;
    
  };
  
};


