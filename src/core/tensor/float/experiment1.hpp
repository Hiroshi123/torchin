



template<typename T>
void Tensor<T>::randBit(){
  
  //std::cout << pow(2,sizeof(T)*12) << std::endl;
  
  double randomSeed = 100;
  std::mt19937 randomEngine(randomSeed);
  std::uniform_real_distribution<double> randomDoubleDistribution(0.0, 1.0);
  
  std::uniform_int_distribution<int> randomIntDistribution(1, pow(2,sizeof(T)*8));
  
  T value;
  
  for(int i = 0 ; i < this->shape.elements ; i++ ){
    
    value = randomIntDistribution(randomEngine);
    this->data[i] = value;
  }
  
  //std::for_each(this->data.begin(),this->data.end(),[](T e){});
  std::cout << this->shape.elements << std::endl;
  
}


template<typename T>
void Tensor<T>::bitFeed(std::vector<T> x){
  
  int count = 0;
  int nwords = sizeof(T) * 8;
  int stock = 0;
  int digit = 0;
  
  for(int i = 0 ; i < x.size() ; i++){
    while(x[i]) {
      stock += (x[i]&1) ? pow(2,digit):0;
      digit ++;x[i]>>=1;
    }
    this->data[count++] = stock;
    stock = 0;digit = 0;
    //x[i]>>=1; 
  }
};


template<typename T>
void Tensor<T>::transpose(int shrinkIndex){
  
  for(int i = 0 ; i < 10 ; i++){
    std::cout << this->data[i] << std::endl;
  }
  
  //this->print("dimensions");
  
  std::cout << this->shape.elements << std::endl;
  //std::cout << sizeof(T)*8 << std::endl;
  
  std::cout << "dim " << this->shape.dimensions[1] << std::endl;
  
  //for(int i = 0 ; i < a->shape.elements ; i++ ){
  
  uint32_t p = 1;
  //T s = 0;
  uint32_t s = 0;
  int z = 0;
  int unit = sizeof(uint32_t)*8;
  int c = 0;
  int column = this->shape.dimensions[0];
  int row    = this->shape.dimensions[1];
  
  uint32_t temp_ptr[this->shape.elements];
  for(int i = 0; i < this->shape.elements; i++)
    temp_ptr[i] = this->data[i];
  
  for(int f = 0 ; f < column ; f++ ){
    p = 1;
    for(int i = 0 ; i < unit ;i++){
      s = 0;
      //this loop is for iteration inside of shranked index.
      for(int j = 0 ; j < row ; j++ ){
	//this loop is for addition of components on different rows
	s += ((p&temp_ptr[f+column*j])==0)? 0:(1<<(j%unit));
	// if you come to the limit of type, you will put an accumulated value on new tensor
	if((j+1) % unit == 0 || j == row - 1){
	  this->data[c++] = s;
	  s = 0;
	}
      }
      
      //std::cout << s << std::endl;
      p <<= 1;
    }
  }

  //a->data = temp_ptr*;
  
  std::cout << c << "  " << this->shape.elements << std::endl;
  
  //a->data = temp_ptr;
  
  for(int i = 0 ; i < 10 ; i++){
    std::cout << this->data[i] << std::endl;
  }
  
  
  if(this->shape.dimensions[1] % unit == 0){
    const std::vector<int> shape = {this->shape.dimensions[1] / unit ,this->shape.dimensions[0] * unit};
    //this->reshape(shape);
  } else {
    const std::vector<int> shape = {this->shape.dimensions[1] / unit + 1 ,this->shape.dimensions[0] * unit};
    this->reshape(shape);
  }
  
};

