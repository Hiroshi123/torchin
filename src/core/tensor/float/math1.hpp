

template<typename T>
constexpr T Tensor<T>::sum(){
  T ever = 0;
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    ever += this->data[i];
  }
  return ever;
};

template<typename T>
constexpr T Tensor<T>::mean(){
  
  T som = this->sum();
  som /= this->shape.elements;
  return som;
  
};


template<typename T>
constexpr T Tensor<T>::max(){
  
  T maxD = this->data[0];
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    if(maxD < this->data[i] )
      maxD = this->data[i];
  }
  
  return maxD;  
};

template<typename T>
constexpr T Tensor<T>::min(){
  
  T minD = this->data[0];
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    if(minD > this->data[i] )
      minD = this->data[i];
  }
  
  return minD;
};

template<typename T>
constexpr void Tensor<T>::pow(size_t power){
  
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    this->data[i] = std::pow(this->data[i],power);
  }
  
};

template<typename T>
constexpr void Tensor<T>::pow(const size_t power, const Tensor<T>& out){
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    out.data[i] = std::pow(this->data[i],power);
  }
};


template<typename T>
constexpr void Tensor<T>::log(const size_t base){
  
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    this->data[i] = std::log(this->data[i]) / std::log(base);
  }
  
};

template<typename T>
constexpr void Tensor<T>::log(const size_t base,const Tensor<T>& out){
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    out.data[i] = std::log(this->data[i]) / std::log(base);
  }
};

template<typename T>
constexpr void Tensor<T>::exp(const size_t power){
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    this->data[i] = std::exp(this->data[i]);
  }
};

template<typename T>
constexpr void Tensor<T>::exp(const size_t base,const Tensor<T>& out){
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    out.data[i] = std::exp(this->data[i]);
  }
};

template<>
void Tensor<float>::rand(float min , float max){
  
  //double randomSeed = 100;
  //melcenn
  //std::mt19937 randomEngine(randomSeed);
  
  //uniform distribution
  std::default_random_engine generator;
  std::uniform_real_distribution<float> randomFloatDistribution(min, max);
  
  //non deterministric random generationnn
  //std::random_device rnd;
  
  for (size_t i = 0; i < this->shape.elements ; ++i) {
    //std::cout << rand100(mt) << "\n";
    //std::cout << norm(mt) << "\n";
    this->data[i] = randomFloatDistribution(generator);
  }
  
};



