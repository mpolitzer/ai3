#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#define SIGMOID_P	0.5
#define LEARNING_RATE	1.0

#define NUM_IN	2
#define NUM_MID	10
#define NUM_OUT	1

float vin[NUM_IN];
float vmid[NUM_MID];
float vout[NUM_OUT];
float expected[NUM_OUT];

float emid[NUM_MID];
float eout[NUM_OUT];

float wmid[NUM_IN][NUM_MID];
float wout[NUM_MID][NUM_OUT];

static float sigmoid(float h)
{
	return 1.0/(1.0 + exp(-h/SIGMOID_P));
}

static void init(void)
{
	int i;
	float *ptr;
	
	for(i = 0, ptr = (float *)&wmid[0]; i < NUM_IN * NUM_MID; i++) {
		*ptr++ = ((float)rand())/UINT_MAX;
	}

	for(i = 0, ptr = (float *)&wout[0]; i < NUM_MID * NUM_OUT; i++) {
		*ptr++ = ((float)rand())/UINT_MAX;
	}
}

static void update_values(void)
{
	int j;

	for(j = 0; j < NUM_MID; j++) {
		int i;

		for(i = 0, vmid[j] = 0; i < NUM_IN; i++) {
			vmid[j] += vin[i] * wmid[i][j];
		}
		vmid[j] = sigmoid(vmid[j]);
	}

	for(j = 0; j < NUM_OUT; j++) {
		int i;

		for(i = 0, vout[j] = 0; i < NUM_MID; i++) {
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
		}
	}

	for(j = 0; j < NUM_IN; j++) {
		int i;

		for(i = 0; i < NUM_MID; i++) {
			wmid[j][i] += LEARNING_RATE
				* emid[i]
				* vin[j];
		}
	}
}

int main(int argc, const char *argv[])
{
	init();

	while (scanf(" %f %f %f", &vin[0], &vin[1], &expected[0]) == 3) {
		update_values();
		update_error();
		update_weights();
	}

	vin[1] = 0;
	vin[0] = 0;
	update_values();
	printf("(0,0) %f\n", vout[0]);

	vin[1] = 0;
	vin[0] = 1;
	update_values();
	printf("(0,1) %f\n", vout[0]);

	vin[1] = 1;
	vin[0] = 0;
	update_values();
	printf("(1,0) %f\n", vout[0]);

	vin[1] = 1;
	vin[0] = 1;
	update_values();
	printf("(1,1) %f\n", vout[0]);

	return 0;
}
