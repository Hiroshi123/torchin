

#include "container.hpp"

namespace nn
{
  
  //class Sequential : public nn::Container {
  
  //template<typename T>
  
  class Parallel : public nn::Module {
    
    std::vector<std::unique_ptr<nn::Module> > seq;
    
    //std::vector<nn::Module*> seq;
    
  public:
    
    Parallel() = default;
    
    // ~Sequential(){
    //   for(size_t i = 0 ; i < seq.size() ; i++){
    // 	delete seq[i];
    //   }
    // };
    
    void name() const;
    
    //void forward();
    
    void add(nn::Module* module);
    
    void insert(nn::Module* module,size_t index);
    void remove(nn::Module* module,size_t index);
    size_t size() const;
    
    //template<typename T>
    Tensor<>& updateOutput(Tensor<>&);
    Tensor<>& updateGradInput(Tensor<>&);
    
  };
  

};



