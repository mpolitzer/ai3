#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define LEARNING_RATE	0.2

#define NUM_IN	2
#define NUM_MID	3
#define NUM_OUT	1

float vin[NUM_IN];
float vmid[NUM_MID];
float vout[NUM_OUT];
float expected[NUM_OUT];

float emid[NUM_MID];
float eout[NUM_OUT];

float mid_offset[NUM_MID];
float out_offset[NUM_OUT];
float wmid[NUM_IN][NUM_MID];
float wout[NUM_MID][NUM_OUT];

static float sigmoid(float h)
{
	return 1.0/(1.0 + exp(-h));
}

static void init(void)
{
	int i;
	float *ptr;

	srand(time(NULL));
	
	for(i = 0, ptr = (float *)&wmid[0]; i < NUM_IN * NUM_MID; i++) {
		*ptr++ = ((float)rand())/UINT_MAX - 0.5;
	}

	for(i = 0, ptr = (float *)&wout[0]; i < NUM_MID * NUM_OUT; i++) {
		*ptr++ = ((float)rand())/UINT_MAX - 0.5;
	}
}

static void update_values(void)
{
	int j;

	for(j = 0; j < NUM_MID; j++) {
		int i;

		for(i = 0, vmid[j] = mid_offset[j]; i < NUM_IN; i++) {
			vmid[j] += vin[i] * wmid[i][j];
		}
		vmid[j] = sigmoid(vmid[j]);
	}

	for(j = 0; j < NUM_OUT; j++) {
		int i;

		for(i = 0, vout[j] = out_offset[j]; i < NUM_MID; i++) {
			vout[j] += vmid[i] * wout[i][j];
		}
		vout[j] = sigmoid(vout[j]);
	}
}

static void update_error(void)
{
	int j, i;

	for(j = 0; j < NUM_OUT; j++) {
		eout[j] = vout[j]
			* (1.0 - vout[j])
			* (expected[j] - vout[j]);
	}

	for(i = 0; i < NUM_IN; i++) {
		int j;

		for(j = 0, emid[i] = 0; j < NUM_MID; j++) {
			emid[i] += vmid[i]
				* (1.0 - vmid[i])
				* (wmid[i][j] * eout[j]);
		}
	}
}

static void update_weights(void)
{	
	int j;

	for(j = 0; j < NUM_MID; j++) {
		int i;

		for(i = 0; i < NUM_OUT; i++) {
			wout[j][i] += LEARNING_RATE
				* eout[i]
				* vmid[j];
			out_offset[i] += LEARNING_RATE * eout[i];
		}
	}

	for(j = 0; j < NUM_IN; j++) {
		int i;

		for(i = 0; i < NUM_MID; i++) {
			wmid[j][i] += LEARNING_RATE
				* emid[i]
				* vin[j];
			mid_offset[i] += LEARNING_RATE * emid[i];
		}
	}
}

static float calc(float v0, float v1)
{
	vin[1] = v1;
	vin[0] = v0;
	update_values();

	return vout[0];
}

static void dump_net(void)
{
	int i;

	printf("\n");
	for(i = 0; i < NUM_IN; i++) {
		int j;

		for(j = 0; j < NUM_MID; j++) {
			printf("%f, ", wmid[i][j]);
		}
		printf("\n");
	}
	printf("---------------\n");
	for(i = 0; i < NUM_OUT; i++)
		printf("%f ", out_offset[i]);
	printf("\n");
	for(i = 0; i < NUM_MID; i++)
		printf("%f ", mid_offset[i]);
	printf("\n");
}

static void dump(void)
{
	float v00 = calc(0,0),
	      v10 = calc(0,1),
	      v01 = calc(1,0),
	      v11 = calc(1,1);

	printf("(0,0) %f, (1,0) %f, (0,1) %f (1,1) %f\n",
			v00, v10, v01, v11);
}

int main(int argc, const char *argv[])
{
	init();

	while (scanf(" %f %f %f", &vin[0], &vin[1], &expected[0]) == 3) {
		update_values();
		update_error();
		update_weights();
		dump();
	}
	dump_net();
	return 0;
}
