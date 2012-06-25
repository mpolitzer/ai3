#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>
#include <time.h>

#define PRINT_INTERVAL 5

#define ITERATIONS 1000000
#define LEARNING_RATE	0.8

#define NUM_IN	5
#define NUM_MID	20
#define NUM_OUT	1

long double vin[NUM_IN];
long double vmid[NUM_MID];
long double vout[NUM_OUT];
long double expected[NUM_OUT];

long double emid[NUM_MID];
long double eout[NUM_OUT];

long double mid_offset[NUM_MID];
long double out_offset[NUM_OUT];
long double wmid[NUM_IN][NUM_MID];
long double wout[NUM_MID][NUM_OUT];

static long double sigmoid(long double h)
{
	return 1.0/(1.0 + exp(-h));
}

static void init(void)
{
	int i;
	long double *ptr;

	srand(time(NULL));
	
	for(i = 0, ptr = (long double *)&wmid[0]; i < NUM_IN * NUM_MID; i++) {
		*ptr++ = ((long double)rand())/UINT_MAX - 0.5;
	}

	for(i = 0, ptr = (long double *)&wout[0]; i < NUM_MID * NUM_OUT; i++) {
		*ptr++ = ((long double)rand())/UINT_MAX - 0.5;
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

	for(i = 0; i < NUM_MID; i++) {
		int j;

		for(j = 0, emid[i] = 0; j < NUM_OUT; j++) {

			emid[i] += vmid[i]
				* (1.0 - vmid[i])
				* (wout[i][j] * eout[j]);
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
			printf("%Lf, ", wmid[i][j]);
		}
		printf("\n");
	}
	for(i = 0; i < NUM_MID; i++) {
		int j;

		for(j = 0; j < NUM_OUT; j++) {
			printf("%Lf, ", wout[i][j]);
		}
		printf("\n");
	}
	for(i = 0; i < NUM_MID; i++)
		printf("%Lf ", mid_offset[i]);
	printf("\n");
	for(i = 0; i < NUM_OUT; i++)
		printf("%Lf ", out_offset[i]);
	printf("\n");
}

void cputf(long double f, int bg)
{
	printf("\033[%dm%LE\033[0m ", bg, f);
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
	long train_set;
	long test_set;
	long lines_in_fin;
	long iteration = 0;
	long correct = 0, wrong = 0;

	FILE *fin;

	clock_t now;
	int elapsed;
	int interval = PRINT_INTERVAL;

	long double age, gender, number_of_tweets, result_past_time, category, result;

	if (argc < 2) {
		fprintf(stderr, "Need filename as parameter\n");
		return 1;
	}
	fin = fopen(argv[1], "r");
	
	init();

	now = clock();

	if (fscanf(fin, " %ld", &lines_in_fin) != 1) return 1;

	train_set = 0.8*lines_in_fin;
	while (train_set > iteration) {
		int ret = fscanf(fin, " %LE %LE %LE %LE %LE %LE",
					&age, &gender, &number_of_tweets,
					&result_past_time, &category, &result);
		if (ret != 6) break;
		vin[0] = age;
		vin[1] = gender;
		vin[2] = number_of_tweets;
		vin[3] = result_past_time;
		vin[4] = category;
		expected[0] = result;

		iteration++;
		elapsed = (clock()-now)/CLOCKS_PER_SEC;
		if (elapsed >= interval) {
			interval += PRINT_INTERVAL;
			printf("iteration number: %ld\n", iteration);
			dump_net();
		}

		update_values();
		update_error();
		update_weights();
	}

	printf("iteration number: %ld\n", iteration);
	dump_net();

	test_set = lines_in_fin - iteration;
	iteration = 0;

	while (iteration < test_set) {
		int ret = fscanf(fin, " %LE %LE %LE %LE %LE %LE",
					&age, &gender, &number_of_tweets,
					&result_past_time, &category, &result);
		if (ret != 6) break;
		vin[0] = age;
		vin[1] = gender;
		vin[2] = number_of_tweets;
		vin[2] = result_past_time;
		vin[4] = category;

		iteration++;
		elapsed = (clock()-now)/CLOCKS_PER_SEC;
		if (elapsed >= interval) {
			interval += PRINT_INTERVAL;
			printf("iteration number: %ld\n", iteration);
		}
		update_values();
		if (vout[0] > 0.5) vout[0] = 1;
		else vout[0] = 0;

		if (vout[0] == result) correct++;
		else wrong++;
	}

	printf("correct: %Lf wrong: %Lf, total: %ld\n",
			((long double)correct)/test_set,
			((long double)wrong)/test_set,
			test_set);
#if 0
	printf("---------------\n");
	printf("dump:\n");
	printf("---------------\n");

	for (i=0; i<10; i++) {
		vin[0] = (i & 1) ? 1 : 0;
		vin[1] = (i & 2) ? 1 : 0;
		update_values();
		dump();
	}
#endif
	return 0;
}
