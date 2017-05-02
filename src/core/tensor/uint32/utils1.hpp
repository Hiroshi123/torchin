
template<>
void Tensor<uint32_t>::set(){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++ ){
    this->data[i] |= (i == this->shape.typeElements - 1 ) ?
      ((1 << this->shape.elements % (sizeof(uint32_t) * 8)) - 1 ) : 0xffffffff;
  }
};



template<>
void Tensor<uint32_t>::set(const std::vector<size_t>& v){
  
  size_t stock1 = 1;
  size_t stock2 = 0;
  
  for(int i = v.size() - 1 ; i >= 0 ; i--){
    stock2 += v[i] * stock1;
    stock1 *= this->shape.dimensions[i]; 
  }
  
  size_t index1 = stock2 / ( sizeof(uint32_t) * 8 );
  size_t index2 = stock2 % ( sizeof(uint32_t) * 8 );
  
  this->data[index1] |= (1 << index2);
  
};

template<>
void Tensor<uint32_t>::unset(){

  for(size_t i = 0 ; i < this->shape.typeElements ; i++ ){
    this->data[i] &= 0x00000000;
  }
  
};

template<>
void Tensor<uint32_t>::unset(const std::vector<size_t>& v){
  
  //std::cout << v[0] << " " << v[1] << std::endl;
  
  size_t stock1 = 1;
  size_t stock2 = 0;
  
  for(int i = v.size() - 1 ; i >= 0 ; i--){
    stock2 += v[i] * stock1;
    stock1 *= this->shape.dimensions[i]; 
  }
  
  size_t index1 = stock2 / ( sizeof(uint32_t) * 8 );
  size_t index2 = stock2 % ( sizeof(uint32_t) * 8 );
  
  this->data[index1] &= ~(1 << index2);
  
};

template<>
const char* Tensor<uint32_t>::get(const std::vector<size_t>& v) const {
  
  //std::cout << v[0] << " " << v[1] << std::endl;
  
  size_t stock1 = 1;
  size_t stock2 = 0;
  
  for(int i = v.size() - 1 ; i >= 0 ; i--){
    stock2 += v[i] * stock1;
    stock1 *= this->shape.dimensions[i]; 
  }
  
  
  size_t index1 = stock2 / ( sizeof(uint32_t) * 8 );
  size_t index2 = stock2 % ( sizeof(uint32_t) * 8 );
  
  char const *ret = ( this->data[index1] & (1 << index2) )?  "1" : "0";
  
  return ret;
};


template<>
void Tensor<uint32_t>::rand(float min,float max){
  
  std::default_random_engine generator;
  std::uniform_real_distribution<float> randomFloatDistribution(min, max);
  
  //non deterministric random generation
  //std::random_device rnd;
  
  for (size_t i = 0; i < this->shape.typeElements ; ++i) {
    
    for(size_t j = 0 ; j < sizeof(uint32_t) * 8 ; j++){
      
      this->data[i] += ( randomFloatDistribution(generator) > 0.5 )? (1 << j) : 0 ;
      
      if (i * sizeof(uint32_t) * 8 + j == this->shape.elements - 1 ){
	//std::cout << "hei!" << std::endl;
	break;
      }
	
    }
  }
  
};

// template<>
// uint32_t Tensor<uint32_t>::count(uint32_t i)
// {
//   i = i - ((i >> 1) & 0x55555555);
//   i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
//   return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
// };

uint32_t count(uint32_t i)
{
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
};


template<>
void Tensor<uint32_t>::dot(const Tensor<uint32_t>& t,const Tensor<uint32_t>& out,std::string type){
  
  size_t d = ((type == "RC") || (type == "CC"))? t.shape.dimensions[0] : t.shape.dimensions[1];
  size_t outC = ((type == "RC") || (type == "CC"))? t.shape.dimensions[1] : t.shape.dimensions[0];
  size_t outR = ((type == "RR") || (type == "RC"))? this->shape.dimensions[0] : this->shape.dimensions[1];
  
  d /= sizeof(uint32_t) * 8; 
  
  // std::cout << t.shape.dimensions[1] << "," << t.shape.dimensions[0] << "," << this->shape.dimensions[1] << std::endl;
  // std::cout << d << "," << outC << "," << outR << std::endl;
  
  uint32_t temp = 0;
  
  //#ifdef _OPENMP
  //#pragma omp parallel for
  //#endif
  
  size_t shiftR1 ,shiftL1 = 0;
  size_t shiftR2 ,shiftL2 = 0;
  
  for(size_t i = 0 ; i < outC ; i++ ){
    for(size_t j = 0 ; j < outR ; j++ ){
      for(size_t k = 0 ; k <= d ; k++ ){
	
	shiftR1 = ((k == 0) & (shiftL1 != 0))? sizeof(uint32_t) * 8 - shiftL1 : 0;
	shiftR2 = ((k == 0) & (shiftL2 != 0))? sizeof(uint32_t) * 8 - shiftL2 : 0;
	
	shiftL1 = (k == d)? sizeof(uint32_t) * 8 - ( k % (sizeof(uint32_t) * 8) ) : 0;
	shiftL2 = (k == d)? sizeof(uint32_t) * 8 - ( k % (sizeof(uint32_t) * 8) ) : 0;
	
	// if (k == d) {
	//   if(shiftV1 != 0)
	//     shiftV1 = sizeof(uint32_t) * 8 - ( k % sizeof(uint32_t) * 8 );
	//   if(shiftV2 != 0)
	//     shiftV2 = sizeof(uint32_t) * 8 - ( k % sizeof(uint32_t) * 8 );
	// }
	
	temp += count(~(((this->data[j*d+k] << shiftR1) >> shiftL1) ^ ((t.data[i*d+k] << shiftR2) >> shiftL2)));
	
	//out.data[i*outR+j] 
	
	//temp += this->data[j*d+k] * t.data[i*d+k];
	//out.data[i*outR+j] = temp;temp = 0;
	
      }
      out.data[i*outR+j] = temp ; temp = 0;
    }
  }
};

template<>
void Tensor<uint32_t>::copy(const Tensor<uint32_t>& out){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    out.data[i] = this->data[i];
  }
  
};


