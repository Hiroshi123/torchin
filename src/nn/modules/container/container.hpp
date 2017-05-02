

#include <vector>
#include "module.hpp"
#include <algorithm>

namespace nn {
  
  class Container : public nn::Module{    
    
  public:
    
    //std::vector<nn::Module*> seq;
    
    Container();
    
    // virtual void add(nn::Module module);
    // virtual void insert(nn::Module module,size_t index);
    // virtual void remove(nn::Module module,size_t index);
    // virtual size_t size();
    // virtual void training();
    // virtual void evaluate();
    
    
  };
  
  Container::Container(){
    
  };
  
  // void Container::add(nn::Module module){
    
  //   seq.push_back(&module);
    
  //   //seq.push_back(&&module);
    
  // };
  
  // void Container::insert(nn::Module module,size_t index){
  //   seq.insert(seq.begin()+index, &module);
  // };
  
  // void Container::remove(nn::Module module,size_t index){
  //   seq.erase(seq.begin(), seq.begin()+index);
  // };
  
  // size_t Container::size(){
  //   return seq.size();
  // };
  
  // void Container::training(){
    
  // };

  // void Container::evaluate(){
    
  //   // std::transform(seq.begin(), seq.end(), seq.begin(),
  //   // 		   [](nn::Module *m) {return new nn::Module(m);});
    
  //   std::for_each(seq.begin(), seq.end(),
  // 		  [](nn::Module* s) { std::cout << "1" << std::endl; });
  
  // };
  
  
};
