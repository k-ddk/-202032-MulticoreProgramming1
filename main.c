#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"

const char* CLASS_NAME[] = {  //사진? 이름들..
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck"
};

void print_usage_and_exit(char** argv) {  //명령행 인자를 받아서 
	fprintf(stderr, "Usage: %s <number of image> <output>\n", argv[0]);  //
	fprintf(stderr, " e.g., %s 3000 result.out\n", argv[0]);
	exit(EXIT_FAILURE);
}

void* read_bytes(const char* fn, size_t n) {  //내가 열어봐야 할 cifar10_image.bin이랑 총 크기??3000*IMAGE_CHW를 받아서
	FILE* f = fopen(fn, "rb");  //파일 열기
	void* bytes = malloc(n);  //void형 포인터 bytes에 n만큼의 공간을 할당
	size_t r = fread(bytes, 1, n, f);  //bytes를 버퍼로 f로 연 파일에서 n*1바이트 데이터 읽어오기
	fclose(f);  //cifar10파일 다 읽었으니까 파일 닫기..
	if (r != n) {  //파일의 내용을 다 읽어오지 못한 경우
		fprintf(stderr,
			"%s: %zd bytes are expected, but %zd bytes are read.\n",
			fn, n, r);
		exit(EXIT_FAILURE);
	}
	return bytes;  //읽어온 내용이 담긴 버퍼를 반환
}

/*
 * Read images from "cifar10_image.bin".
 * CIFAR-10 test dataset consists of 10000 images with (3, 32, 32) size.
 * Thus, 10000 * 3 * 32 * 32 * sizeof(float) = 122880000 bytes are expected.
 */
const int IMAGE_CHW = 3 * 32 * 32 * sizeof(float);  //*이미지 한 개의 크기가 32*32 고 그때에 rgb 3개씩 더 곱한 것. 즉, 이게 cifar10 파일안 데이터의 사이즈
float* read_images(size_t n) {  //인자 n장의 이미지를 읽는 함수
	return (float*)read_bytes("cifar10_image.bin", n * IMAGE_CHW);  //이때 이미지 한장마다 사이즈는 32x32, 에다가 rgb채널 3개 
	//이미지 한장에 대해 32x32x3 개의 바이트가 존재
}

/*
 * Read labels from "cifar10_label.bin".
 * 10000 * sizeof(int) = 40000 bytes are expected.
 */
int* read_labels(size_t n) {
	return (int*)read_bytes("cifar10_label.bin", n * sizeof(int));
}

/*
 * Read network from "network.bin".  <-이 파일에 weight들이 아래와 같이 저장되어 있음
 * conv1_1 : weight ( 64,   3, 3, 3) bias ( 64)
 * conv1_2 : weight ( 64,  64, 3, 3) bias ( 64)
 * conv2_1 : weight (128,  64, 3, 3) bias (128)
 * conv2_2 : weight (128, 128, 3, 3) bias (128)
 * conv3_1 : weight (256, 128, 3, 3) bias (256)
 * conv3_2 : weight (256, 256, 3, 3) bias (256)
 * conv3_3 : weight (256, 256, 3, 3) bias (256)
 * conv4_1 : weight (512, 256, 3, 3) bias (512)
 * conv4_2 : weight (512, 512, 3, 3) bias (512)
 * conv4_3 : weight (512, 512, 3, 3) bias (512)
 * conv5_1 : weight (512, 512, 3, 3) bias (512)
 * conv5_2 : weight (512, 512, 3, 3) bias (512)
 * conv5_3 : weight (512, 512, 3, 3) bias (512)
 * fc1     : weight (512, 512) bias (512)
 * fc2     : weight (512, 512) bias (512)
 * fc3     : weight ( 10, 512) bias ( 10)
 * Thus, 60980520 bytes are expected.
 */
const int NETWORK_SIZES[] = {
	64 * 3 * 3 * 3, 64,
	64 * 64 * 3 * 3, 64,
	128 * 64 * 3 * 3, 128,
	128 * 128 * 3 * 3, 128,
	256 * 128 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256,
	512 * 256 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512, 512,
	512 * 512, 512,
	10 * 512, 10
};

float* read_network() {
	return (float*)read_bytes("network.bin", 60980520);  //네트워크빈파일(가중치와 bias가 담긴 파일) 열어서 안의 내용을 float형 배열 안에 넣어서 리턴..
}

float** slice_network(float* p) {  //가중치랑 bias 배열..
	float** r = (float**)malloc(sizeof(float*) * 32);  //?????이차포인터 ..32개???
	for (int i = 0; i < 32; ++i) {  //이미지 크기 가 32...
		r[i] = p;  //p는 network내용이 일차배열로 담겨있는거
		p += NETWORK_SIZES[i];  //p의 값이 갈수록 증가함...
	}
	return r;  //배열을 반환.. 네트워크 사이즈 값들을 누적해서 더해서 나온 배열..  r은 32x60980520 크기임..
}

int main(int argc, char** argv) {
	if (argc != 3) {  //인자가 세개가 아닐 경우
		print_usage_and_exit(argv);
	}

	int num_images = atoi(argv[1]);  //내가 인자로 넣어둔 3000이 들어감
	float* images = read_images(num_images);  //float형 배열 images, 3000을 보냄, cifar10 내용이 담긴 버퍼를 리턴
	float* network = read_network();  //network.bin 파일을 읽어서 버퍼에 담고. 가중치와 bias담긴 배열 반환
	float** network_sliced = slice_network(network);  //!!!!!!!!!!
	int* labels = (int*)calloc(num_images, sizeof(int));  //int형..3000만큼 공간 할당 해주고 , 3000개의 이미지를 볼거임
	float* confidences = (float*)calloc(num_images, sizeof(float));  //int형..3000만큼 공간 할당 해주고 , 3000개의 이미지를 볼거임
	time_t start;  //시간설정해두기 

	printf("OpenCL_CNN\tImages: %4d\n", num_images);  //이미지 갯수 출력하고
	cnn_init();  //!!!!!!!!!!cnn.h를 포함해 줬기 때문에 ~, 시퀀스버전에서는 아무것도 수행되지 않음
	start = clock();  //시간측정 시작
	printf("cnn들어감\n");
	cnn(images, network_sliced, labels, confidences, num_images);  //여기에 이제 kernel.cl에 내가 커널코드짠 거를 넣어서 시간을 측정??..
	printf("cnn나옴\n");
	printf("\tExecution time: %f sec\n", (double)(clock() - start) / CLK_TCK);

	FILE* of = fopen(argv[2], "w");
	int* labels_ans = read_labels(num_images);
	double acc = 0;
	for (int i = 0; i < num_images; ++i) {
		fprintf(of, "Image %04d: %s %f\n", i, CLASS_NAME[labels[i]], confidences[i]);
		if (labels[i] == labels_ans[i]) ++acc;
	}
	fprintf(of, "Accuracy: %f\n", acc / num_images);
	fclose(of);

	printf("\tAccuracy: %f\n", acc / num_images);
	compare(argv[2]);

	free(images);
	free(network);
	free(network_sliced);
	free(labels);
	free(confidences);
	free(labels_ans);

	return 0;
}