#include<quadrature.h>
#include<iostream>
#include<iomanip>
int main(int argc, char* argv[])
{
  unsigned int N = atoi(argv[1]);
  double* cpts = (double*) malloc(N * sizeof(double));
  double* cwts = (double*) malloc(N * sizeof(double)); 
  double a,b; a = -5; b = 5;
  
  clencurt(cpts, cwts, a, b, N);
  
  std::cout << "Points:\n";
  for (unsigned int i = 0; i < N; ++i)
  {
    std::cout << std::setprecision(16) << cpts[i] << std::endl;
  }

  std::cout << "Weights:\n";
  for (unsigned int i = 0; i < N; ++i)
  {
    std::cout << std::setprecision(16) << cwts[i] << std::endl;
  } 

  free(cpts);
  free(cwts);
  return 0;
}
