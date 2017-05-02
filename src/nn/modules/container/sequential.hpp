
#include "container.hpp"

namespace nn
{
  
  //class Sequential : public nn::Container {
  
  //template<typename T>
  
  class Sequential : public nn::Module {
    
    std::vector<std::unique_ptr<nn::Module> > seq;
    
    //std::vector<nn::Module*> seq;
    
  public:
    
    Sequential() = default;
    
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
  
  //Sequential::Sequential() = default;
  
  //template<typename T>
  void Sequential::add(nn::Module* module){
    
    //seq.push_back(module);
    seq.emplace_back(module);
    
  };

  //template<typename T>
  void Sequential::name() const {
    
    std::cout << seq.size() << std::endl;
    
    for(size_t i = 0 ; i < seq.size() ; i++ ){
      seq[i]->name();
    }
    
  };

  //template<typename T>
  void Sequential::insert(nn::Module* module,size_t index){
    //seq.insert(seq.begin()+index, module);
    seq.emplace(seq.begin()+index, module);
  };
  
  //template<typename T>
  void Sequential::remove(nn::Module* module,size_t index){
    seq.erase(seq.begin(), seq.begin()+index);
  };
  
  size_t Sequential::size() const{
    return seq.size();
  };
  
  //template<typename T>
  Tensor<>& Sequential::updateOutput(Tensor<>& input){

    //std::cout << "what a fuck!!" << std::endl;
    
    //seq[0]->updateOutput(input);
    
    for(size_t i = 0 ; i < seq.size() ; i++){
      
      seq[i]->output = (i == 0)?
      	std::move(seq[i]->updateOutput(input)) :
      	std::move(seq[i]->updateOutput(seq[i-1]->output));
      
      //std::cout << i << " layerr" << std::endl;
      //seq[i]->output.print();
      
    }
    
    return seq[seq.size()-1]->output;
  };
  
  //template<typename T>
  Tensor<>& Sequential::updateGradInput(Tensor<>& input){
    
    for(int i = seq.size()-1 ; i >= 0 ; i--){
      
      seq[i]->output = (i == (int) seq.size()-1)?
      	std::move(seq[i]->updateGradInput(input)) :
      	std::move(seq[i]->updateGradInput(seq[i+1]->output));
      
      seq[i]->accGradParameters();
      
    }
    
    return input;
    
  };
  
};

