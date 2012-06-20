#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>

#define ITERATIONS 2000000
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
			int k;

			for(k = 0; k < NUM_OUT; k++) {
				emid[i] += vmid[i]
					* (1.0 - vmid[i])
					* (wmid[i][j] * eout[k]);
			}
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

static void dump_net(void)
{
	int i;

	printf("---------------\n");
	printf("network:\n");
	printf("---------------\n");
	for(i = 0; i < NUM_IN; i++) {
		int j;

		for(j = 0; j < NUM_MID; j++) {
			printf("%f, ", wmid[i][j]);
		}
		printf("\n");
	}
	for(i = 0; i < NUM_MID; i++) {
		int j;

		for(j = 0; j < NUM_OUT; j++) {
			printf("%f, ", wout[i][j]);
		}
		printf("\n");
	}
	for(i = 0; i < NUM_MID; i++)
		printf("%f ", mid_offset[i]);
	printf("\n");
	for(i = 0; i < NUM_OUT; i++)
		printf("%f ", out_offset[i]);
	printf("\n");
}

void cputf(float f, int bg)
{
	printf("\033[%dm%f\033[0m ", bg, f);
}

static void dump(void)
{
	int i;

	for (i = 0; i < NUM_IN; i++) {
		cputf(vin[i], vin[i] > 0.5 ? 42 : 41);
	}
	printf(" : ");

	for (i = 0; i < NUM_OUT; i++) {
		cputf(vout[i], vout[i] > 0.5 ? 42 : 41);
	}
	printf("\n");
}

void gen_expected_not(void)
{
	unsigned int v1 = vin[0];
	expected[0] = (~v1) & 1;
}

void gen_expected_xor(void)
{
	unsigned int v1 = vin[0], v2 = vin[1];
	expected[0] = (v1&1) ^ (v2&1);
}

int main(int argc, const char *argv[])
{
	int i, n;

	if (argc > 1) sscanf(argv[1], " %d", &n);
	else n = ITERATIONS;

	init();

	do {
		vin[0] = (n & 1) ? 1 : 0;
		vin[1] = (n & 2) ? 1 : 0;
		gen_expected_xor();

		update_values();
		update_error();
		update_weights();
	} while (n--);

	dump_net();

	printf("---------------\n");
	printf("dump:\n");
	printf("---------------\n");

	for (i=0; i<4; i++) {
		vin[0] = (i & 1) ? 1 : 0;
		vin[1] = (i & 2) ? 1 : 0;
		update_values();
		dump();
	}

	return 0;
}
