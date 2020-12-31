#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cnn.h"

static void pooling2x2(float* input, float* output, int N) {  //이거는 컨볼루션에 비해 간단하니까 쉽게 할 수 있을 거다~
	int i, j, k, l;
	for (i = 0; i < N; i++) {  //전체 이미지?도는거
		for (j = 0; j < N; j++) {
			float max = 0;
			for (k = 0; k < 2; k++) {  //2x2짜리니까 필터도는거
				for (l = 0; l < 2; l++) {
					float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
					max = (max > pixel) ? max : pixel;  //최대값으로
				}
			}
			output[i * N + j] = max;  //output에 넣어주기
		}
	}
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(float* inputs, float* outputs, int D, int N) {  //인접한 픽셀값 중 가장 큰 값을 선택하는 것
	int i;
	for (i = 0; i < D; i++) {  //채널만큼 돌아가면서 이미지 한 장에 대해서 한 부분...
		float* input = inputs + i * N * N * 4;  //input & output에 대한 오프셋을 잡아주고
		float* output = outputs + i * N * N;
		pooling2x2(input, output, N);  //실질적인? 함수에 넣음
	}
}

static void convolution3x3(float* input, float* output, float* filter, int N) {
	int i, j, k, l;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {  //위의 N과 이 N은 이미지 사이즈를 말함
			float sum = 0;
			for (k = 0; k < 3; k++) {  //k와 l은 필터의 x축, y축을 가리킴
				for (l = 0; l < 3; l++) {
					int x = i + k - 1;  //x와 y의 인덱스를 설정해 주고..
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N)  //제로패딩된 부분에 대해 실제로 제로패딩한 부분에 대해서는 연산을 수행하지 않는다는 부분 이거 고려해서 코드 짤 것
						sum += input[x * N + y] * filter[k * 3 + l];  //제로패딩을 하거나~ 안 하거나~ 빠른 걸로.. 이런 순차 코드를 병렬화 해라..
				}
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
#define ReLU(x) (((x)>0)?(x):0)
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {
	int i, j;

	memset(outputs, 0, sizeof(float) * N * N * D2);

	//컨볼루션 수행 연산
	for (j = 0; j < D2; j++) {  //아웃풋 채널에 대한 for문
		for (i = 0; i < D1; i++) {  //인풋 채널에 대한 for문
			float* input = inputs + N * N * i;  //포인터의 오프셋 이동중..
			float* output = outputs + N * N * j;
			float* filter = filters + 3 * 3 * (j * D1 + i);
			convolution3x3(input, output, filter, N);  //오프셋이 이동된 것들을 컨볼루션 수행 함수에 넣어줌
		}  //convolution3x3 수행하는 부분에 신경써서 해라... 이 부분이 시간이 많이 걸릴거다...
	}

	//부차적인 연산들, 이건 간단하니까 설명 생략하고 위의 컨볼루션 먼저 해보겠다 
	for (i = 0; i < D2; i++) {
		float* output = outputs + N * N * i;
		float bias = biases[i];
		for (j = 0; j < N * N; j++) {
			output[j] = ReLU(output[j] + bias);  //ReLU는 0보다 작은 값들은 0으로 하고..
		}  //define돼 있으니까 병렬화 하기쉬울거다~
	}
}

/*
 * M = output size  뒤에있는 layer
 * N = input size  앞에 있는 layer
 */
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {  //fully connected layer 소스코드...
	int i, j;
	for (j = 0; j < M; j++) {  //뒤에 있는 레이어를 돌면서
		float sum = 0;
		for (i = 0; i < N; i++) {
			sum += input_neuron[i] * weights[j * N + i];  //누적합구하는 중
		}
		sum += biases[j];  //마지막으로 bias를 더해주고
		output_neuron[j] = ReLU(sum);  //ReLU 해주기
	}
}

static void softmax(float* output, int N) {  //output값에 대한 함수 N은 10...
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {  //output값 중 최대값 구하기
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);  //2의 저 인자수만큼의 승..
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float* fc, int N) {  //fc3랑 10을 받음
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

void cnn_init() {
	// nothing to init in the sequential version
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	// slice the network into weights and biases
	//난 이것들이 뭔지 모르겠어..ㄴ
	float* w1_1, *b1_1, *w1_2, *b1_2;
	float* w2_1, *b2_1, *w2_2, *b2_2;
	float* w3_1, *b3_1, *w3_2, *b3_2, *w3_3, *b3_3;
	float* w4_1, *b4_1, *w4_2, *b4_2, *w4_3, *b4_3;
	float* w5_1, *b5_1, *w5_2, *b5_2, *w5_3, *b5_3;
	float* w1, *b1, *w2, *b2, *w3, *b3;
	w1_1 = network[0]; b1_1 = network[1];  //main.c에 있는 NETWORK_SIZES 에서 0 은 weight, 1은 bias..
	w1_2 = network[2]; b1_2 = network[3];  //한줄 한줄을 담음..
	w2_1 = network[4]; b2_1 = network[5];
	w2_2 = network[6]; b2_2 = network[7];
	w3_1 = network[8]; b3_1 = network[9];
	w3_2 = network[10]; b3_2 = network[11];
	w3_3 = network[12]; b3_3 = network[13];
	w4_1 = network[14]; b4_1 = network[15];
	w4_2 = network[16]; b4_2 = network[17];
	w4_3 = network[18]; b4_3 = network[19];
	w5_1 = network[20]; b5_1 = network[21];
	w5_2 = network[22]; b5_2 = network[23];
	w5_3 = network[24]; b5_3 = network[25];
	w1 = network[26]; b1 = network[27];
	w2 = network[28]; b2 = network[29];
	w3 = network[30]; b3 = network[31];

	// allocate memory for output of each layer
	float* c1_1, *c1_2, *p1;
	float* c2_1, *c2_2, *p2;
	float* c3_1, *c3_2, *c3_3, *p3;
	float* c4_1, *c4_2, *c4_3, *p4;
	float* c5_1, *c5_2, *c5_3, *p5;
	float* fc1, *fc2, *fc3;
	c1_1 = alloc_layer(64 * 32 * 32);  //이만큼의 공간을 할당해 줌
	c1_2 = alloc_layer(64 * 32 * 32);
	p1 = alloc_layer(64 * 16 * 16);
	c2_1 = alloc_layer(128 * 16 * 16);
	c2_2 = alloc_layer(128 * 16 * 16);
	p2 = alloc_layer(128 * 8 * 8);
	c3_1 = alloc_layer(256 * 8 * 8);
	c3_2 = alloc_layer(256 * 8 * 8);
	c3_3 = alloc_layer(256 * 8 * 8);
	p3 = alloc_layer(256 * 4 * 4);
	c4_1 = alloc_layer(512 * 4 * 4);
	c4_2 = alloc_layer(512 * 4 * 4);
	c4_3 = alloc_layer(512 * 4 * 4);
	p4 = alloc_layer(512 * 2 * 2);
	c5_1 = alloc_layer(512 * 2 * 2);
	c5_2 = alloc_layer(512 * 2 * 2);
	c5_3 = alloc_layer(512 * 2 * 2);
	p5 = alloc_layer(512 * 1 * 1);
	fc1 = alloc_layer(512);
	fc2 = alloc_layer(512);
	fc3 = alloc_layer(10);

	// run network
	for (int i = 0; i < num_images; ++i)
	{
		float* image = images + i * 3 * 32 * 32;                               //다음 이미지로 넘어감

		convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);               //컨볼루션 풀링 컨볼루션 풀링~ 이런식인데
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);              //여기를 최적화 해주면 된다...
		pooling_layer(c1_2, p1, 64, 16);                                //D는 점점 커지고 N은 점점 작아짐

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);  //output 크기 16*16*128
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);  //output 크기 16*16*128  똑같은걸 두번 넣어보는 건가
		pooling_layer(c2_2, p2, 128, 8);

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);  //output 크기 8*8*256
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
		pooling_layer(c3_3, p3, 256, 4);

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);  //output크기 4*4*512
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
		pooling_layer(c4_3, p4, 512, 2);

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);  //output크기 2*2*512
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		pooling_layer(c5_3, p5, 512, 1);

		fc_layer(p5, fc1, w1, b1, 512, 512);
		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		softmax(fc3, 10)

			labels[i] = find_max(fc3, 10);  //라벨을 채우고~
		confidences[i] = fc3[labels[i]];  //컨피던스 채우고~
	}

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);
}

//하기 쉬운 것 부터 차근차근 for문 풀어나가면서 해라