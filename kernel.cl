
__kernel void convolution3x3(__global float* input, __global float* output, __global float* filter, int N) 
{
	int k, l;

	int i = get_global_id(1);
	int j = get_global_id(0);
	__local float sum;

	if (i<N&&j<N)
    {  
	    sum=0;
		for (k = 0; k < 3; k++) 
		{  //k와 l은 필터의 x축, y축을 가리킴
			for (l = 0; l < 3; l++) 
			{
				int x = i + k - 1;  //x와 y의 인덱스를 설정해 주고..
				int y = j + l - 1;
				if (x >= 0 && x < N && y >= 0 && y < N)  //제로패딩된 부분 연산수행 x
					sum += input[x * N + y] * filter[k * 3 + l];  
			}
			output[i * N + j] += sum;
		}
	}
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
__kernel void kernel_convolution_layer(__global float* inputs, __global float* outputs, __global float* filters, __global float* biases, int D2, int D1, int N) 
{

	int j = get_global_id(1);
	int i = get_global_id(0);
	__global float* input;
	__global float* output1;
    __global float* output2;
	__global float* filter;


    if(j<D2&&i<D1) {  
			input = inputs + N * N * i;  //포인터의 오프셋 이동중..
			output1 = outputs + N * N * j;

			 filter = filters + 3 * 3 * (j * D1 + i);
			 convolution3x3(input, output1, filter, N);  //오프셋이 이동된 것들을 컨볼루션 수행 함수에 넣어줌
	}


	int k = get_global_id(1);
	int l = get_global_id(0);
	if (k<D2&&l<N*N)
	{
		    output2 = outputs + N * N * k;

			 output2[l] = (output2[l]+biases[k])>0 ? (output2[l]+biases[k]):0;  
	}	
}
__kernel void pooling2x2(__global float* input, __global float* output, int N) 
{  
	int i = get_global_id(1);
	int j  = get_global_id(0);
	int k, l;

	float max = 0;
	for (k = 0; k < 2; k++) {  //2x2 필터
		for (l = 0; l < 2; l++) {
			float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
			max = (max > pixel) ? max : pixel;  //최대값으로
		}
	}
	output[i * N + j] = max;  //output에 넣어주기
} 
/*__kernel void pooling_layer(__global float* inputs, __global float* outputs, int D, int N) 
{ 
	int i = get_global_id(0);

	__global float* input = inputs + i * N * N * 4;  //input & output에 대한 오프셋을 잡아주고
	__global float* output = outputs + i * N * N;
	pooling2x2(input, output, N);  //실질적인? 함수에 넣음
}*/