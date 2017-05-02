
// template<typename T>
// const T Tensor<T>::operator[](const std::vector<size_t>& v) const{
  
//   assert ( v.size() == this->shape.dimensions.size() );
  
//   size_t index = 0;
//   for(int i = this->shape.dimensions.size()-1 ; i >= 0 ; i--){
//     assert(v[i] <= this->shape.dimensions[i]);
//     if(i < (int)this->shape.dimensions.size()-1){
//       index += v[i]* this->shape.dimensions[i+1];
//     } else {
//       index += v[i];
//     }
//   }
  
//   std::cout <<  "debug " << std::endl;
  
//   return this->data[index];
  
// };


template<>
float& Tensor<float>::operator[](const std::vector<size_t> v){
  
  /*
    boundary check 
  */
  
  assert ( v.size() == this->shape.dimensions.size() );
  
  size_t index = 0;
  
  for(int i = this->shape.dimensions.size()-1 ; i >= 0 ; i--){
    assert(v[i] <= this->shape.dimensions[i]);
    if(i < (int) this->shape.dimensions.size()-1){
      index += v[i] * this->shape.dimensions[i+1];
    } else {
      index += v[i];
    }
  }
  
  return this->data[index];
  
};





