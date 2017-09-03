/*************************************************/
/* bp1.c                                         */
/* Neural network learning with back propagation */
/*************************************************/

#define IMAGE_FILE "train-images-idx3-ubyte"
#define LABEL_FILE "train-labels-idx1-ubyte"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>

#ifdef GPERF
#include <gperftools/profiler.h>
#endif // GPERF

#include "./fix16.h"

#define INPUTNO  (28*28)    // No of input cell
#define OUTPUTNO (10)
#define HIDDENNO (50)    // No of hidden cell
#define ALPHA    (10)   // Coefficient of learning
#define SEED     (65535)  // Seed of random
#define MAXINPUTNO (60000)  // Max number of learning data
#define BATCH_SIZE (1)
#define LEARNING_RATE (0.1)
#define WEIGHT_INIT (0.01)

int open_image ();
int open_label ();

int getdata (int fd_image,
			 int fd_label,
			 fix16_t *in_data,
			 int     *ans);
void hlearn (fix16_t **wh, // weight of hidden layer
			 fix16_t *wo, // weight of output layer
			 int output_size, int input_size,
			 fix16_t *hi, // current hidden layer ansewr
			 fix16_t *e,  // input value
			 fix16_t *o);   // output value
void print_images (fix16_t data[INPUTNO], int label);
void olearn (fix16_t *wo,
			 int     element_size,
			 fix16_t *hi,
			 fix16_t *e,
			 fix16_t *ans,
			 fix16_t *o);

fix16_t affine (const int output_size,
                const int input_size,
                const int batch_size,
                fix16_t *out,           // [batch_size][output_size],
                const fix16_t *in_data, // [batch_size][input_size],
                const fix16_t *wh,      // [input_size][output_size],
                const fix16_t *wb);      // [output_size]
fix16_t affine_backward (const int output_size,
                         const int hidden_size,
                         const int batch_size,
                         fix16_t *dx,  // [batch_size][output_size],
                         fix16_t *db,  // [output_size],
                         fix16_t *dw,  // [output_size][hidden_size],
                         const fix16_t *dout, // [batch_size][output_size],
                         const fix16_t *w,    // [hidden_size][output_size],
                         const fix16_t *x);   // [batch_size][hidden_size],

void relu (const int batch_size,
		   const int size,
		   fix16_t *o,         // [batch_size][size],
		   const fix16_t *e);  // [batch_size][size]
fix16_t relu_backward (const int batch_size,
                       const int size,
                       fix16_t       *dx,  // [batch_size][size],
                       const fix16_t *x,   // [batch_size][size],
                       const fix16_t *dout); // [batch_size][size]);

fix16_t softmax (const int batch_size,
                 const int size,
                 fix16_t       *o,  // [batch_size][size],
                 const fix16_t *e); // [batch_size][size]
fix16_t softmax_backward (const int batch_size,
                          const int size,
                          fix16_t       *dx,   // [batch_size][size],
                          const fix16_t *y,    // [batch_size][size],
                          const fix16_t *t);   // [batch_size][size])

void forward (fix16_t *o, fix16_t *e, int input_size, int output_size);

void TestNetwork (const int input_size,
				  const int output_size,
				  const int hidden_size,
				  const fix16_t *wh0,   // [input_size][hidden_size],
				  const fix16_t *wb0,   // [hidden_size],
				  const fix16_t *wh1,   // [hidden_size][output_size],
				  const fix16_t *wb1);   // [output_size]

int argmax (const int x_size, fix16_t *o);

void initwh(const int y_size, const int x_size, fix16_t *wh);
void initwb(const int x_size, fix16_t *wb);

double rand_normal ( double mu, double sigma );
double drnd ();

const char *message = "hello\r\n";

// extern char _binary_train_images_idx3_ubyte_100_start[];
// extern char _binary_train_images_idx3_ubyte_100_end[];

extern char _binary_t10k_images_idx3_ubyte_start[];
extern char _binary_t10k_images_idx3_ubyte_end[];

extern char _binary_t10k_labels_idx1_ubyte_start[];
extern char _binary_t10k_labels_idx1_ubyte_end[];

extern char _binary_wb0_bin_start[];
extern char _binary_wb0_bin_end[];
extern char _binary_wb1_bin_start[];
extern char _binary_wb1_bin_end[];
extern char _binary_wh0_bin_start[];
extern char _binary_wh0_bin_end[];
extern char _binary_wh1_bin_start[];
extern char _binary_wh1_bin_end[];


const char* hex_enum[] = {"0", "1", "2", "3", "4", "5", "6", "7",
                          "8", "9", "a", "b", "c", "d", "e", "f"};
int main ()
{
  write (STDOUT_FILENO, message, strlen (message));

  int i;

  int len = _binary_t10k_images_idx3_ubyte_end - _binary_t10k_images_idx3_ubyte_start;

  const fix16_t *wh0 = (fix16_t *)_binary_wh0_bin_start;  // [INPUTNO * HIDDENNO];
  const fix16_t *wb0 = (fix16_t *)_binary_wb0_bin_start;  // [HIDDENNO];
  const fix16_t *wh1 = (fix16_t *)_binary_wh1_bin_start;  // [HIDDENNO * OUTPUTNO];
  const fix16_t *wb1 = (fix16_t *)_binary_wb1_bin_start;  // [OUTPUTNO];

  TestNetwork (INPUTNO, OUTPUTNO, HIDDENNO, wh0, wb0, wh1, wb1);

  return 0;
}

fix16_t af0 [BATCH_SIZE * HIDDENNO];
fix16_t fix16_in_data[INPUTNO];
char *in_data;
char *ans_data;
fix16_t af1 [BATCH_SIZE * OUTPUTNO];
fix16_t rel0[BATCH_SIZE * HIDDENNO];
fix16_t rel1[BATCH_SIZE * OUTPUTNO];

void TestNetwork (const int input_size,
				  const int output_size,
				  const int hidden_size,
				  const fix16_t *wh0,  // [input_size][hidden_size],
				  const fix16_t *wb0,  // [hidden_size],
				  const fix16_t *wh1,  // [hidden_size][output_size],
				  const fix16_t *wb1)  // [output_size]
{
  const char *message0 = "=== TestNetwork ===\n";
  write (STDOUT_FILENO, message0, strlen (message0));

  in_data  = &_binary_t10k_images_idx3_ubyte_start[0x10];
  ans_data = &_binary_t10k_labels_idx1_ubyte_start[0x08];

  // write (STDOUT_FILENO, hex_enum[(in_data[0] >> 12) & 0x0f], 2);
  // write (STDOUT_FILENO, hex_enum[(in_data[0] >>  8) & 0x0f], 2);
  // write (STDOUT_FILENO, hex_enum[(in_data[0] >>  4) & 0x0f], 2);
  // write (STDOUT_FILENO, hex_enum[(in_data[0] >>  0) & 0x0f], 2);

  int correct = 0;

  for (int no_input = 0; no_input < 100; no_input += BATCH_SIZE) {
    write (STDOUT_FILENO, hex_enum[((no_input) >> 12) & 0x0f], 2);
    write (STDOUT_FILENO, hex_enum[((no_input) >>  8) & 0x0f], 2);
    write (STDOUT_FILENO, hex_enum[((no_input) >>  4) & 0x0f], 2);
    write (STDOUT_FILENO, hex_enum[((no_input) >>  0) & 0x0f], 2);
	write (STDOUT_FILENO, "\r\n", sizeof ("\r\n"));

	// for (int b = 0; b < BATCH_SIZE; b++) {
	//   uint8_t image[INPUTNO];
	//   // if (read (image_fd, image, INPUTNO * sizeof(unsigned char)) == -1)  { perror("read"); exit (EXIT_FAILURE); }
	//   for (int i = 0; i < INPUTNO; i++) {
	// 	in_data[b][i] = fix16_from_dbl (image[i] / 255.0);
	//   }
	//   uint8_t label;
	//   // if (read (label_fd, &label, sizeof(uint8_t)) == -1) { perror("read"); exit (EXIT_FAILURE); }
	//   ans_data[b] = label;
	// }
	//

	for (int i = 0; i < 28 * 28; i++) {
	  char hex_value = in_data[i];

	  write (STDOUT_FILENO, hex_enum[(hex_value >> 4) & 0x0f], 2);
	  write (STDOUT_FILENO, hex_enum[(hex_value >> 0) & 0x0f], 2);

	  fix16_in_data[i] = fix16_from_dbl (in_data[i] / 255.0);

	  if ((i % 28) == 27) { write (STDOUT_FILENO, "\r\n", 2); }
	}

	affine (HIDDENNO, INPUTNO,  BATCH_SIZE, af0, (const fix16_t *)fix16_in_data, wh0, wb0);
	relu (BATCH_SIZE, HIDDENNO, rel0, af0);
	affine (OUTPUTNO, HIDDENNO, BATCH_SIZE, af1, rel0,    wh1, wb1);

	for (int b = 0; b < BATCH_SIZE; b++) {
	  int t = argmax (OUTPUTNO, &af1[b * OUTPUTNO]);
	  if (t == ans_data[b]) correct++;
	  printf ("Ans_Data = %d, Label = %d\n", t, ans_data[b]);
	}

	in_data += INPUTNO;
	ans_data ++;
  }
  printf ("Correct = %d\n", correct);

  const char *message1 = "Correct = ";
  write (STDOUT_FILENO, message1, strlen (message1));

  write (STDOUT_FILENO, hex_enum[(correct >> 4) & 0x0f], 2);
  write (STDOUT_FILENO, hex_enum[(correct >> 0) & 0x0f], 2);

  return;
}



fix16_t affine (const int output_size,
			   const int input_size,
			   const int batch_size,
			   fix16_t *out,            // [batch_size][output_size],
			   const fix16_t *in_data,  // [batch_size][input_size],
			   const fix16_t *wh,       // [input_size][output_size],
			   const fix16_t *wb)       // [output_size]
{
  for (int b = 0; b < batch_size; b++) {
  	for (int o = 0; o < output_size; o++) {
  	  out[b * output_size + o] = fix16_from_dbl (0.0);
  	  for (int i = 0; i < input_size; i++) {
  	  	out[b * output_size + o] = fix16_add (out[b * output_size + o],
                                              fix16_mul (in_data[b * input_size + i], wh[i * output_size + o]));
  	  }
  	  out[b * output_size + o] = fix16_add (out[b * output_size + o], wb[o]);
  	}
  }
}


fix16_t affine_backward (const int output_size,
						const int hidden_size,
						const int batch_size,
						fix16_t *dx,  // [batch_size][hidden_size],
						fix16_t *db,  // [output_size],
						fix16_t *dw,  // [hidden_size][output_size],
						const fix16_t *dout,  // [batch_size][output_size],
						const fix16_t *w,     // [hidden_size][output_size],
						const fix16_t *x)     // [batch_size][hidden_size],
{
  for (int b = 0; b < batch_size; b++) {
	for (int h = 0; h < hidden_size; h++) {
	  dx[b * hidden_size + h] = 0;
	  for (int o = 0;o < output_size; o++) {
		dx[b * hidden_size + h] = fix16_add (dx[b * hidden_size + h],
                                             fix16_mul (dout[b * output_size + o], w[h * output_size + o]));
	  }
	}
  }
  for (int h = 0; h < hidden_size; h++) {
	for (int o = 0; o < output_size; o++) {
	  dw[h * output_size + o] = 0;
	  for (int b = 0; b < batch_size; b++) {
		dw[h * output_size + o] = fix16_add (dw[h * output_size + o],
                                             fix16_mul (x[b * hidden_size + h], dout[b * output_size + o]));
	  }
	}
  }

  for (int o = 0; o < output_size; o++) {
	db[o] = 0;
	for (int b = 0; b < batch_size; b++) {
	  db[o] = fix16_add (db[o], dout[b * output_size + o]);
	}
  }
}


void relu (const int batch_size,
		   const int size,
		   fix16_t *o,        // [batch_size][size],
		   const fix16_t *e)  // [batch_size][size]
{
  for (int b = 0; b < batch_size; b++) {
	for (int i = 0; i < size; i++) {
	  o[b * size + i] = e[b * size + i] > 0 ? e[b * size + i] : 0;
	}
  }
  return;
}


fix16_t relu_backward (const int batch_size,
					  const int size,
					  fix16_t       *dx,     // [batch_size][size],
					  const fix16_t *x,      // [batch_size][size],
					  const fix16_t *dout)   // [batch_size][size])
{
  for (int b = 0; b < batch_size; b++) {
	for (int i = 0; i < size; i++) {
	  dx[b * size + i] = x[b * size + i] > 0 ? dout[b * size + i] : 0;
	}
  }
}


fix16_t softmax (const int batch_size,
                 const int size,
                 fix16_t *o,       // [batch_size][size],
                 const fix16_t *e) // e[batch_size][size]
{
  fix16_t *max = (fix16_t *)malloc(sizeof(fix16_t) * batch_size);
  for (int b = 0; b < batch_size; b++) {
	max[b] = e[b * size + 0];
	for (int i = 1; i < size; i++) {
	  max[b] = max[b] < e[b * size + i] ? e[b * size + i] : max[b];
	}
  }

  for (int b = 0; b < batch_size; b++) {
	fix16_t exp_sum;
    exp_sum = 0;
	fix16_t *a = (fix16_t *)malloc(sizeof(fix16_t) * size);
	for (int i = 0; i < size; i++) {
      a[i] = fix16_sub (e[b * size + i], max[b]);
	  exp_sum = fix16_add (exp_sum, fix16_exp(a[i]));
	}
	for (int i = 0; i < size; i++) {
	  o[b * size + i] = fix16_div (fix16_exp(a[i]), exp_sum);
	}
    free (a);
  }


  free (max);
}


fix16_t softmax_backward (const int batch_size,
                          const int size,
                          fix16_t        *dx,  // [batch_size][size],
                          const fix16_t  *y,  // [batch_size][size],
                          const fix16_t  *t)  // [batch_size][size]
{
  for (int b = 0; b < batch_size; b++) {
	for (int y_idx = 0; y_idx < size; y_idx++) {
	  dx[b * size + y_idx] = fix16_div (fix16_sub (y[b * size + y_idx], t[b * size + y_idx]),
                                        fix16_from_int (batch_size));
	}
  }
}


int argmax (const int x_size, fix16_t *o)
{
  fix16_t ret = o[0];
  int    max_idx = 0;
  for (int i = 1; i < x_size; i++) {
	if (o[i] > ret) {
	  ret = o[i];
	  max_idx = i;
	}
  }

  return max_idx;
}


void initwh (const int y_size, const int x_size, fix16_t *wh)
{
  for (int y = 0; y < y_size; y++) {
	for (int x = 0; x < x_size; x++) {
	  wh[y * x_size + x] = fix16_from_dbl (WEIGHT_INIT * rand_normal (0.0, 1.0));
	}
  }
}


void initwb (const int x_size, fix16_t *wb)
{
  for (int j = 0; j < x_size + 1; j++) {
    wb[j] = 0;
  }
}


double rand_normal (double mu, double sigma)
{
  // double z = sqrt( -2.0 * log(drnd()) ) * sin( 2.0 * M_PI * drnd() );
  // return mu + sigma*z;
  return drnd ();
}


double drnd ()
{
  double rndno;
  while ((rndno = (double)rand() / RAND_MAX) == 1.0);
  return rndno;
}
