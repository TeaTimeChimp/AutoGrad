#include "Test.h"
#include "Tensor.h"
#include "Test_Distribution.h"
#include "Test_Linear.h"
#include "Test_Embedding.h"
#include "Test_Matrix.h"
#include "Test_NDArray.h"
#include "Test_Tensor.h"


using namespace std;


void Assert(const bool pass,const char* const msg)
{
	if(!pass)
		throw TestFailed(msg);
}


void print(const std::string& str)
{
	cout<<str<<endl;
}


void print(const TensorPtr& data,const char* label)
{
	if(label)
		cout<<label<<endl;
	data->Print();
	cout<<endl;
}


void print(const NDArray& data,const char* label)
{
	if(label)
		cout<<label<<endl;
	data.Print(cout);
	cout<<endl;
}


void print(const NDShape& v)
{
	cout<<"[";
	bool first = true;
	for(auto i:v)
	{
		if(!first)
			cout<<",";
		cout<<i;
		first = false;
	}
	cout<<"]"<<endl;
}


void Test()
{
	Test_NDArray();
	Test_Tensor();
	Test_Linear();
	Test_Embedding();
	Test_Distribution();
}