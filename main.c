#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"

const char* CLASS_NAME[] = {  //����? �̸���..
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

void print_usage_and_exit(char** argv) {  //����� ���ڸ� �޾Ƽ� 
	fprintf(stderr, "Usage: %s <number of image> <output>\n", argv[0]);  //
	fprintf(stderr, " e.g., %s 3000 result.out\n", argv[0]);
	exit(EXIT_FAILURE);
}

void* read_bytes(const char* fn, size_t n) {  //���� ������� �� cifar10_image.bin�̶� �� ũ��??3000*IMAGE_CHW�� �޾Ƽ�
	FILE* f = fopen(fn, "rb");  //���� ����
	void* bytes = malloc(n);  //void�� ������ bytes�� n��ŭ�� ������ �Ҵ�
	size_t r = fread(bytes, 1, n, f);  //bytes�� ���۷� f�� �� ���Ͽ��� n*1����Ʈ ������ �о����
	fclose(f);  //cifar10���� �� �о����ϱ� ���� �ݱ�..
	if (r != n) {  //������ ������ �� �о���� ���� ���
		fprintf(stderr,
			"%s: %zd bytes are expected, but %zd bytes are read.\n",
			fn, n, r);
		exit(EXIT_FAILURE);
	}
	return bytes;  //�о�� ������ ��� ���۸� ��ȯ
}

/*
 * Read images from "cifar10_image.bin".
 * CIFAR-10 test dataset consists of 10000 images with (3, 32, 32) size.
 * Thus, 10000 * 3 * 32 * 32 * sizeof(float) = 122880000 bytes are expected.
 */
const int IMAGE_CHW = 3 * 32 * 32 * sizeof(float);  //*�̹��� �� ���� ũ�Ⱑ 32*32 �� �׶��� rgb 3���� �� ���� ��. ��, �̰� cifar10 ���Ͼ� �������� ������
float* read_images(size_t n) {  //���� n���� �̹����� �д� �Լ�
	return (float*)read_bytes("cifar10_image.bin", n * IMAGE_CHW);  //�̶� �̹��� ���帶�� ������� 32x32, ���ٰ� rgbä�� 3�� 
	//�̹��� ���忡 ���� 32x32x3 ���� ����Ʈ�� ����
}

/*
 * Read labels from "cifar10_label.bin".
 * 10000 * sizeof(int) = 40000 bytes are expected.
 */
int* read_labels(size_t n) {
	return (int*)read_bytes("cifar10_label.bin", n * sizeof(int));
}

/*
 * Read network from "network.bin".  <-�� ���Ͽ� weight���� �Ʒ��� ���� ����Ǿ� ����
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
	return (float*)read_bytes("network.bin", 60980520);  //��Ʈ��ũ������(����ġ�� bias�� ��� ����) ��� ���� ������ float�� �迭 �ȿ� �־ ����..
}

float** slice_network(float* p) {  //����ġ�� bias �迭..
	float** r = (float**)malloc(sizeof(float*) * 32);  //?????���������� ..32��???
	for (int i = 0; i < 32; ++i) {  //�̹��� ũ�� �� 32...
		r[i] = p;  //p�� network������ �����迭�� ����ִ°�
		p += NETWORK_SIZES[i];  //p�� ���� ������ ������...
	}
	return r;  //�迭�� ��ȯ.. ��Ʈ��ũ ������ ������ �����ؼ� ���ؼ� ���� �迭..  r�� 32x60980520 ũ����..
}

int main(int argc, char** argv) {
	if (argc != 3) {  //���ڰ� ������ �ƴ� ���
		print_usage_and_exit(argv);
	}

	int num_images = atoi(argv[1]);  //���� ���ڷ� �־�� 3000�� ��
	float* images = read_images(num_images);  //float�� �迭 images, 3000�� ����, cifar10 ������ ��� ���۸� ����
	float* network = read_network();  //network.bin ������ �о ���ۿ� ���. ����ġ�� bias��� �迭 ��ȯ
	float** network_sliced = slice_network(network);  //!!!!!!!!!!
	int* labels = (int*)calloc(num_images, sizeof(int));  //int��..3000��ŭ ���� �Ҵ� ���ְ� , 3000���� �̹����� ������
	float* confidences = (float*)calloc(num_images, sizeof(float));  //int��..3000��ŭ ���� �Ҵ� ���ְ� , 3000���� �̹����� ������
	time_t start;  //�ð������صα� 

	printf("OpenCL_CNN\tImages: %4d\n", num_images);  //�̹��� ���� ����ϰ�
	cnn_init();  //!!!!!!!!!!cnn.h�� ������ ��� ������ ~, ���������������� �ƹ��͵� ������� ����
	start = clock();  //�ð����� ����
	printf("cnn��\n");
	cnn(images, network_sliced, labels, confidences, num_images);  //���⿡ ���� kernel.cl�� ���� Ŀ���ڵ�§ �Ÿ� �־ �ð��� ����??..
	printf("cnn����\n");
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