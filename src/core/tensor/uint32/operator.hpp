

template<>
std::string Tensor<uint32_t>::operator[](const std::vector<size_t>& v) {
  
  size_t stock1 = 1;
  size_t stock2 = 0;
  
  for(int i = v.size() - 1 ; i >= 0 ; i--){
    stock2 += v[i] * stock1;
    stock1 *= this->shape.dimensions[i]; 
  }
  
  size_t index1 = stock2 / ( sizeof(uint32_t) * 8 );
  size_t index2 = stock2 % ( sizeof(uint32_t) * 8 );
  
  std::cout << "ans " << ( this->data[index1] & (1 << index2) ) << std::endl;
  
  std::string a = "";
  
  if( this->data[index1] & (1 << index2) ){
    a = "1";
  } else {
    a = "0";
  }
  
  return a;
  
};


// template<>
// uint32_t& Tensor<uint32_t>::operator[](std::vector<size_t> v){
  
//   /*
//     boundary check 
//   */
  
//   // for( auto &el : v ){
//   //   el -= 1;
//   // }
  
//   std::cout << "size "<< v.size() << std::endl; 
  
//   assert ( v.size() == this->shape.dimensions.size() );
  
//   //size_t index = 0;
  
//   size_t stock1 = 1;
//   size_t stock2 = 0;
  
//   for(int i = v.size() - 1 ; i >= 0 ; i--){
    
//     //std::cout << v[i] * stock1 << std::endl;
//     stock2 += v[i] * stock1;
//     stock1 *= this->shape.dimensions[i];
    
//   }
  
//   size_t index1 = stock2 / ( sizeof(uint32_t) * 8 );
//   size_t index2 = stock2 % ( sizeof(uint32_t) * 8 );
  
//   std::cout << index1 << " " << index2 << std::endl;

//   //std::cout << stock2 << std::endl;
  
  
//   //this->data[index1] |= (1 << index2);
  
//   std::cout << this->data[index1] << std::endl;
  
//   //size_t index1 = std::accumulate(v.begin(),v.end(),1,std::multiplies<size_t>()) / ( sizeof(uint32_t) * 8 );
//   //size_t index2 = std::accumulate(v.begin(),v.end(),1,std::multiplies<size_t>()) % ( sizeof(uint32_t) * 8 );
  
//   //std::cout << index1 << "," << index2 << std::endl;
  
//   // for(int i = this->shape.dimensions.size()-1 ; i >= 0 ; i--){
//   //   assert(v[i] <= this->shape.dimensions[i]);
//   //   if(i < (int) this->shape.dimensions.size()-1){
//   //     index += v[i] * this->shape.dimensions[i+1];
//   //   } else {
//   //     index += v[i];
//   //   }
//   //
  
//   std::cout << "ans " << ( this->data[index1] & (1 << index2)) << std::endl;
  
//   uint32_t a = 0;
  
//   if( this->data[index1] & (1 << index2) ){
//     std::cout << "{0,0} = " << 1 << std::endl;
//   } else{
//     std::cout << "{0,0} = " << 0 << std::endl;
//   }
  
//   return this->data[index1];
//   //return a;
  
// };





