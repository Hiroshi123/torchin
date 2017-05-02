
#include "shape.hpp"

#ifdef _OPENMP
//#pragma omp parallel for
#endif

//typedef uint32_t DEFAULT_TENSOR_TYPE;
typedef float DEFAULT_TENSOR_TYPE;

//template<typename T = uint32_t>
template<typename T = DEFAULT_TENSOR_TYPE>
class Tensor {
  
private:
  
  std::unique_ptr<T[]> data; // (new int[10]);
  
  //std::shared_ptr<T[]> data; // (new int[10]);
  
public:
  
  /* if there is no call for constructor, default constructor would be called. */
  Tensor<T>() = default;
  
  /* otherwise, following constructor would be called */
  
  Tensor(const std::vector<size_t>& vec);
  Tensor(const std::vector<size_t>& vec,size_t index,size_t nb);
  
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  
  /* 
     following ordinary destructor, copy and move are non-necessary 
     because Tensor contains unique_ptr, and the algorithm under it would take the jobs for them.
  */
  
  //~Tensor<T>();
  
  //Tensor<T>(const Tensor<T>&) = delete;
  //Tensor<T>(Tensor<T>&&) = delete;
  //Tensor<T>& operator=(const Tensor<T>&) = delete;
  //Tensor<T>& operator=(const Tensor<T>&&) = default;
  
  Shape shape;
  Shape& get_shape();
  
  void resize(const std::vector<size_t> &dims);
  void reshape(const std::vector<size_t> &dims);
  
  void print();
  constexpr void print(const std::string);
  
  void bitFeed(std::vector<T>);
  
  void rand(float min = 0.0,float max = 1.0);
  void randn();
  void randBit();
  
  
  constexpr void transpose();
  constexpr void transpose(const Tensor<T>& out);
  
  void transpose(int);
  
  //template<typename uint32_t> void Tensor<uint32_t>::transpose(int);
  void fill(T);
  
  /* deep copy */
  void copy(const Tensor<T>&);

  /* logic operation only for uint32_t */
  
  constexpr void operator & (const Tensor<uint32_t>&);
  constexpr void operator | (const Tensor<uint32_t>&);
  constexpr void operator ^ (const Tensor<uint32_t>&);
  constexpr void operator <<(const size_t);
  constexpr void operator >>(const size_t);
  
  /* these logical operations are large captital to avoid overlaps of pre-defined method name */
  
  constexpr void OR  (const Tensor<uint32_t>&,const Tensor<uint32_t>&);
  constexpr void AND (const Tensor<uint32_t>&,const Tensor<uint32_t>&);
  constexpr void XOR (const Tensor<uint32_t>&,const Tensor<uint32_t>&);
  constexpr void NOT (const Tensor<uint32_t>&,const Tensor<uint32_t>&);
  constexpr void RSHIFT (const size_t , const Tensor<uint32_t>&);
  constexpr void LSHIFT (const size_t , const Tensor<uint32_t>&);
  
  //constexpr void FLIP();
  
  void FLIP();
  void FLIP(const Tensor<uint32_t>&);
  
  //constexpr void FLIP(const Tensor<uint32_t>&);
  
  uint32_t popCount();
  
  /* basic arithmetic operation */
  
  constexpr void add(const Tensor<T>&);
  constexpr void add(const Tensor<T>&,Tensor<T>& out);
  constexpr void operator + (const Tensor<T>&);
  
  constexpr void sub(const Tensor<T>&);
  constexpr void sub(const Tensor<T>&,Tensor<T>& out);
  constexpr void operator - (const Tensor<T> &);
  
  constexpr void mul(const Tensor<T>&);
  constexpr void mul(const Tensor<T>&,Tensor<T>& out);
  constexpr void operator * (const Tensor<T> &);
  
  constexpr void div(const Tensor<T>&);
  constexpr void div(const Tensor<T>&,Tensor<T>& out);
  constexpr void operator / (const Tensor<T> &);
  
  constexpr void dot(const Tensor<T>& t2,const Tensor<T>& out,std::string type = "RC");
  
  constexpr void convolution1D(const Tensor<T>& filter,const Tensor<T>& out,const size_t pad = -1 ,const size_t stride = 1);
  constexpr void convolution2D(const Tensor<T>& filter,const Tensor<T>& out,
			       const size_t paddingW = -1,const size_t paddingH = -1,
			       const size_t strideW = 1,const size_t strideH = 1);
  
  constexpr void convolution3D(const Tensor<T>& filter,const Tensor<T>& out,
			       const size_t paddingW = -1,const size_t paddingH = -1,
			       const size_t strideW = 1,const size_t strideH = 1);
  
  constexpr void maxPooling1D(const Tensor<T>& out,const size_t kw,const size_t dw = -1);
  constexpr void maxPooling2D(const Tensor<T>& out,const size_t kw,const size_t kh,
			      const size_t dw = -1, const size_t dh = -1,
			      const size_t padw = 0, const size_t padh = 0
			      );
  
  constexpr void maxPooling2DwithF(const Tensor<T>& out,const size_t kw,const size_t kh,
				   const size_t dw = -1, const size_t dh = -1,
				   const size_t padw = 0, const size_t padh = 0
				   );
  
  /* basic math method */
  
  constexpr T sum();
  constexpr T mean();
  constexpr T max();
  constexpr T min();
  
  constexpr void pow(const size_t power = 2);
  constexpr void pow(const size_t power,const Tensor<T>& out);
  
  constexpr void log(const size_t base  = 2);
  constexpr void log(const size_t base,const Tensor<T>& out);
  
  constexpr void exp(const size_t power );
  constexpr void exp(const size_t power,const Tensor<T>& out);
  
  //void pika(size_t a,Tensor<T>& );
  
  //const float operator[](const std::vector<size_t>& v) const;
  float& operator[](const std::vector<size_t> v);
  
  /* debug for uint32_t */
  std::string operator[](const std::vector<size_t>& v);
  
  /* bit feed for uint32_t */
  
  constexpr void set();
  constexpr void set(const std::vector<size_t>& v);
  constexpr void unset();
  constexpr void unset(const std::vector<size_t>& v);
  
  const char* get(const std::vector<size_t>& v) const;
  
  //T& operator[](const std::vector<size_t> v,int s);
  
  //const T& operator[](const std::vector<size_t>&& v);
  
  //void operator=(const Tensor<T>& obj);
  
  // Tensor<T> operator += (const Tensor<T>& obj){
  //   std::cout << "hei\n";
  // };
  
  // Tensor<uint32_t>& operator += (const size_t a){
    
  //   std::cout << "pika" << std::endl;
    
  //   //this->m_iNumber += Tensor<T>.m_iNumber;
    
  //   return *this;
  // }
  
  // void operator ++()
  // {
    
  //   std::cout << "++\n"; 
  //   //count = count+1;
  // }
  
  
  /* functional method */
  
  constexpr void apply(const std::function<T(T)> functor);
  
  /* alias of apply */
  constexpr void map (const std::function<T(T)> functor);
  
  constexpr void map2(Tensor<T>&,const std::function<T(T,T)> functor);
  
  //int operator[](int x){return this->data[x];};
  
};

template<>
Tensor<float>::Tensor(const std::vector<size_t>& dimensions)
  : shape(dimensions)
{
  
  data.reset(new float[shape.elements]);
  
  std::fill_n(data.get(), shape.elements, float());
  
  //this->print("dimensions");
};

template<>
Tensor<uint32_t>::Tensor(const std::vector<size_t>& dimensions)
  : shape(dimensions,"uint32_t")
{
  
  data.reset(new uint32_t[shape.typeElements]);
  std::fill_n(data.get(), shape.typeElements, uint32_t());
  
  //this->print("dimensions");
};


template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& dimensions,size_t index,size_t nb)
  : shape(dimensions)
{
  
  //using s = typename std::enable_if<std::is_floating_point<T>::value, T>::type;
  
  data.reset(new T[shape.elements]);
  std::fill_n(data.get(), shape.elements, T());
  
};

#include "float/utils1.hpp"
#include "float/operator.hpp"
#include "float/functions.hpp"
#include "float/arithmetic.hpp"
#include "float/convolution.hpp"
#include "float/pooling.hpp"
#include "float/math1.hpp"
#include "float/print.hpp"
#include "float/experiment1.hpp"

#include "uint32/logic.hpp"
#include "uint32/print.hpp"
#include "uint32/utils1.hpp"
#include "uint32/convolution.hpp"


//#include "uint32/operator.hpp"

