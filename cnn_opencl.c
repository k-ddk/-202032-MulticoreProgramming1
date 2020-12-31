//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define _CRT_SECURE_NO_WARNINGS

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cnn.h"


cl_int err;  //에러 체크 변수
cl_platform_id platform = NULL;
cl_device_id device = NULL;
char version[1024];
cl_context context;
cl_command_queue queue;  //디바이스 한개만 할거니까 한개만 생성
char *kernel_source;
size_t kernel_source_size;
cl_program program;
cl_kernel kernel_pooling_layer, kernel_convolution3x3, kernel_convolution_layer, kernel_fc_layer, kernel_softmax, kernel_find_max;
//size_t global_size[2] = { 256,256 };
//size_t local_size[2] = { 16, 16 };
size_t global_size;
size_t local_size;

cl_mem buf_inputs, buf_outputs, buf_filters, buf_biases;
cl_mem buf_inputs1, buf_outputs1;

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }  //에러 확인 함수

char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE *file = fopen("kernel.cl", "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') cnt++;
	}
	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
}

void cnn_init() {
	/*
	 * TODO
	 * Initialize OpenCL objects as global variables. For example,
	 * clGetPlatformIDs(1, &platform, NULL);
	 */


	 //플랫폼정보 얻기
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);
	//GPU 디바이스 정보 얻기
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);  //모든 타입의 디바이스의 개수를 num_devices에 담아옴
	CHECK_ERROR(err);
	err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 1024, version, NULL);
	CHECK_ERROR(err);
	printf("- CL_DEVICE_VERSION\t: %s\n", version);

	//컨텍스트 생성
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	//커맨드큐 생성
	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	//커널코드 가져오기
	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	//프로그램만들기
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);  //프로그램 오브젝트 생성완료
	CHECK_ERROR(err);

	//프로그램빌드
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char *log;

		// Get program build
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		CHECK_ERROR(err);

		// Get build log
		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error : \n%s\n", log);
		free(log);
		exit(0);
	}
	CHECK_ERROR(err);

	//kernel_pooling_layer = clCreateKernel(program, "kernel_pooling_layer", &err);  //프로그램에서 커널 오브젝트로 생성완료
	//CHECK_ERROR(err);
	kernel_convolution_layer = clCreateKernel(program, "kernel_convolution_layer", &err);
	CHECK_ERROR(err);
	//kernel_fc_layer = clCreateKernel(program, "kernel_fc_layer", &err);
	//CHECK_ERROR(err);
	//kernel_softmax = clCreateKernel(program, "kernel_softmax", &err);
	//CHECK_ERROR(err);
	//kernel_find_max = clCreateKernel(program, "kernel_find_max", &err);

};
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
	/*
																		  //printf("풀링1\n");
	buf_inputs1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * D*D*2*N*2*N, NULL, &err);
	buf_outputs1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D*D*N, NULL, &err);
	//printf("풀링1\n");
	err = clEnqueueWriteBuffer(queue, buf_inputs1, CL_FALSE, 0, sizeof(float) *  D*D * 2 * N * 2 * N, inputs, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, buf_outputs1, CL_FALSE, 0, sizeof(float) * D*D*N, outputs, 0, NULL, NULL);
	CHECK_ERROR(err);
	//printf("풀링1\n");
   err = clSetKernelArg(kernel_pooling_layer, 0, sizeof(cl_mem), &inputs);
   CHECK_ERROR(err);
   err = clSetKernelArg(kernel_pooling_layer, 1, sizeof(cl_mem), &outputs);
   CHECK_ERROR(err);
   err = clSetKernelArg(kernel_pooling_layer, 2, sizeof(int), &D);
   CHECK_ERROR(err);
   err = clSetKernelArg(kernel_pooling_layer, 2, sizeof(int), &N);
   CHECK_ERROR(err);

   size_t global_size = D * D * 2 * N * 2 * N;
   size_t local_size = 4;

   clEnqueueNDRangeKernel(queue, kernel_pooling_layer, 2, NULL, global_size, local_size, 0, NULL, NULL);  //커널 실행
   CHECK_ERROR(err);
   //printf("풀링완료1\n");
   err = clEnqueueReadBuffer(queue, buf_outputs1, CL_TRUE, 0, sizeof(int) * D*D*N, outputs, 0, NULL, NULL);  //컨볼루션 결과 가져오기
   CHECK_ERROR(err);
   //printf("풀링결과가져오기\n");


   err = clReleaseMemObject(buf_inputs1);
   CHECK_ERROR(err);
   //printf("릴리즈0");
   err = clReleaseMemObject(buf_outputs1);
   CHECK_ERROR(err);
   //printf("릴리즈0\n");
   */
	int i;
	for (i = 0; i < D; i++) {  //채널만큼 돌아가면서 이미지 한 장에 대해서 한 부분...
		float* input = inputs + i * N * N * 4;  //input & output에 대한 오프셋을 잡아주고
		float* output = outputs + i * N * N;
		pooling2x2(input, output, N);  //실질적인? 함수에 넣음
	}

}

/*
 * M = output size  뒤에있는 layer
 * N = input size  앞에 있는 layer
 */
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {  //fully connected layer 소스코드...
	int i, j;
	for (j = 0; j < M; j++) {
		float sum = 0;
		for (i = 0; i < N; i++) {
			sum += input_neuron[i] * weights[j * N + i];  //누적합구하는 중
		}
		sum += biases[j];  //마지막으로 bias를 더해주고
		output_neuron[j] = sum > 0 ? sum : 0;  //ReLU 해주기
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

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {  //덩어리를 보내주고 그중 몇번째 장!을 선택할 지를 결정하는 함수
	//메모리 오브젝트를 그때 그때 여기서 생성해주기!
	buf_inputs = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * (N*N*D1), NULL, &err);
	buf_outputs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * (N*N*D2), NULL, &err);
	buf_filters = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * (D2*D1 * 3 * 3), NULL, &err);
	buf_biases = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * D2, NULL, &err);
	//printf("메모리오브젝트생성완료\n");


	err = clEnqueueWriteBuffer(queue, buf_inputs, CL_FALSE, 0, sizeof(float) * (N*N*D1), inputs, 0, NULL, NULL);
	CHECK_ERROR(err);
	//printf("여기1 \n");
	err = clEnqueueWriteBuffer(queue, buf_filters, CL_FALSE, 0, sizeof(float) * (D2*D1 * 3 * 3), filters, 0, NULL, NULL);
	CHECK_ERROR(err);
	//printf("여기2 \n");
	err = clEnqueueWriteBuffer(queue, buf_biases, CL_FALSE, 0, sizeof(float) * D2, biases, 0, NULL, NULL);
	CHECK_ERROR(err);
	//printf("여기3 \n");
	//printf("메모리오브젝트에데이터적기완료\n");


	//커널지정해주기
	err = clSetKernelArg(kernel_convolution_layer, 0, sizeof(cl_mem), &buf_inputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_layer, 1, sizeof(cl_mem), &buf_outputs);
	CHECK_ERROR(err);
	//printf("커널지정완료\n");
	err = clSetKernelArg(kernel_convolution_layer, 2, sizeof(cl_mem), &buf_filters);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_layer, 3, sizeof(cl_mem), &buf_biases);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_layer, 4, sizeof(int), &D2);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_layer, 5, sizeof(int), &D1);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_layer, 6, sizeof(int), &N);
	CHECK_ERROR(err);


	size_t global_size = D1;
	size_t local_size = D2;


	clEnqueueNDRangeKernel(queue, kernel_convolution_layer, 1, NULL, &global_size, &local_size, 0, NULL, NULL);  //커널 실행
	CHECK_ERROR(err);
	//printf("실행완료\n");


	err = clEnqueueReadBuffer(queue, buf_outputs, CL_TRUE, 0, sizeof(int) * (N*N*D2), outputs, 0, NULL, NULL);  //컨볼루션 결과 가져오기
	CHECK_ERROR(err);
	//printf("결과값적었고 이제 릴리즈하고빠져나감\n");


	err = clReleaseMemObject(buf_inputs);
	CHECK_ERROR(err);
	//printf("릴리즈1");
	err= clReleaseMemObject(buf_outputs);
	CHECK_ERROR(err);
	//printf("릴리즈2");
	err = clReleaseMemObject(buf_filters);
	CHECK_ERROR(err);
	//printf("릴리즈3");
	err = clReleaseMemObject(buf_biases);
	CHECK_ERROR(err);
	//printf("릴리즈4\n");
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	//printf("들어옴");
	// slice the network into weights and biases
	float* w1_1, *b1_1, *w1_2, *b1_2;
	float* w2_1, *b2_1, *w2_2, *b2_2;
	float* w3_1, *b3_1, *w3_2, *b3_2, *w3_3, *b3_3;
	float* w4_1, *b4_1, *w4_2, *b4_2, *w4_3, *b4_3;
	float* w5_1, *b5_1, *w5_2, *b5_2, *w5_3, *b5_3;
	float* w1, *b1, *w2, *b2, *w3, *b3;
	w1_1 = network[0]; b1_1 = network[1];
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


	/*
	 * TODO
	 * Implement here.
	 * Write classification results to labels and confidences.       
	 * See "cnn_seq.c" if you don't know what to do.
	 */

	// run network
	for (int i = 0; i < num_images; ++i)
	{
		float* image = images + i * 3 * 32 * 32;  //이미지 한장, 이때 채널이 세개니까 총 이미지1장에 대해 채널까지 3개씩

		convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);               
		//printf("두번째 컨볼;루션\n");
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);              
		//printf("세번째컨볼루션\n");
		pooling_layer(c1_2, p1, 64, 16);                                //D는 점점 커지고 N은 점점 작아짐
		//printf("풀링성공!\n");

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);  //output 크기 16*16*128
		//printf("4번째컨볼루션\n");
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);  //output 크기 16*16*128  똑같은걸 두번 넣어보는 건가
		//printf("5번째컨볼루션\n");
		pooling_layer(c2_2, p2, 128, 8);
		//printf("풀링성공!\n");

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);  //output 크기 8*8*256
		//printf("6번째컨볼루션\n");
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
		//printf("7번째컨볼루션\n");
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
		//printf("8번째컨볼루션\n");
		pooling_layer(c3_3, p3, 256, 4);
		//printf("풀링성공!\n");

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);  //output크기 4*4*512
		//printf("9번째컨볼루션\n");
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
		//printf("10번째컨볼루션\n");
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
		//printf("11번째컨볼루션\n");
		pooling_layer(c4_3, p4, 512, 2);
		//printf("풀링성공!\n");

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);  //output크기 2*2*512
		//printf("12번째컨볼루션\n");
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
		//printf("13번째컨볼루션\n");
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		//printf("14번째컨볼루션\n");
		pooling_layer(c5_3, p5, 512, 1);
		//printf("풀링성공!\n");

		fc_layer(p5, fc1, w1, b1, 512, 512);
		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		softmax(fc3, 10);

		labels[i] = find_max(fc3, 10);  
		confidences[i] = fc3[labels[i]];  
	}

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);
}

