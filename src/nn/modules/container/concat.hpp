

//#include "container.hpp"
#include "module.hpp"

namespace nn
{
  
  //class Concat : public nn::Container {
  
  //template<typename T>
  
  class Concat : public nn::Module {
    
    std::vector<std::unique_ptr<nn::Module> > seq;
    
    size_t concatDim;
    
    //std::vector<nn::Module*> seq;
    
  public:
    
    //Concat();// = default;
    
    Concat(const size_t dim);
    
    // ~Concat(){
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

  Concat::Concat(const size_t dim){
    concatDim = dim;
    
  };
  
  
  void Concat::name() const{
    
    std::cout << "nn.Concat {\n";
    std::cout << "  input\n";
    
  };
  
  void Concat::insert(nn::Module* module,size_t index){
    //seq.insert(seq.begin()+index, module);
    seq.emplace(seq.begin()+index, module);
  };
  
  //template<typename T>
  void Concat::remove(nn::Module* module,size_t index){
    seq.erase(seq.begin(), seq.begin()+index);
  };
  
  size_t Concat::size() const{
    return seq.size();
  };
  
  //template<typename T>
  Tensor<>& Concat::updateOutput(Tensor<>& input){
    
    //std::cout << "what a fuck!!" << std::endl;
    
    //seq[0]->updateOutput(input);
    
    //std::for_each(seq.begin(),seq.end(),[](std::unique_ptr<nn::Module> x){std::cout << "";} );
    
    for(auto it = seq.begin() ; it != seq.end() ; ++it )
      (*it)->output = std::move((*it)->updateOutput(input));
    
    
    //std::move((*it)->name();
    
    //std::cout << "/n";
    
    // for(size_t i = 0 ; i < seq.size() ; i++){
      
    //   seq[i]->output = (i == 0)?
    //   	std::move(seq[i]->updateOutput(input)) :
    //   	std::move(seq[i]->updateOutput(seq[i-1]->output));
      
    //   //std::cout << i << " layerr" << std::endl;
    //   //seq[i]->output.print();
      
    // }
    
    return seq[seq.size()-1]->output;
  };
  
  //template<typename T>
  Tensor<>& Concat::updateGradInput(Tensor<>& input){
    
    for(int i = seq.size()-1 ; i >= 0 ; i--){
      
      seq[i]->output = (i == (int) seq.size()-1)?
      	std::move(seq[i]->updateGradInput(input)) :
      	std::move(seq[i]->updateGradInput(seq[i+1]->output));
      
      seq[i]->accGradParameters();
      
    }
    
    return input;
    
  };
  
  
  
};
