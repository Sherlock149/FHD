# include <bits/stdc++.h>
# include <conio.h>
using namespace std;

float calc(float GT, float X)
{
	return (1 - (X/GT));
}
	
	
int main()
{
	float gt,x;
	int n;
	
	cout<<"Manual Accuracy Calculator for Huamn Detector \n";
	
	float sum = 0;
	cout<<"Enter number of frames: ";
	cin>>n;
	cout<<"\n";
	
	for (int i=0;i<n;i++)
	{
		cout<<"Frame no:"<<i<<"\n";
		cout<<"(FP + FN):";
		cin>>x;
		cout<<" GT:";
		cin>>gt;
		sum += calc(gt,x);
	}
	
	cout<<"Accuracy: "<<100*sum/n<<"%";
	
	getch();
	return 0;
		
}
