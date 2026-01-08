/* Includes ---------------------------------------------------------------- */
#include <apnea_detection_inferencing.h>
#include <ESP_I2S.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// I2S object
I2SClass I2S;

// Pin config for ESP32-S3 DevKit + INMP441
#define I2S_SCK 14
#define I2S_WS  15
#define I2S_SD  16

#define LED_PIN 2
#define APNEA_THRESHOLD 0.5

/** Audio inference buffer struct */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];

static bool debug_nn = false;
static bool record_status = true;
static int apnea_count;
static int loop_count;

/* Function prototypes ----------------------------------------------------- */
static bool microphone_inference_start(uint32_t n_samples);
static bool microphone_inference_record(void);
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr);
static void capture_samples(void* arg);
static void microphone_inference_end(void);

/**
 * @brief Arduino setup
 */
void setup()
{
    Serial.begin(115200);
    while (!Serial);

    Serial.println("Edge Impulse Inferencing Demo with ESP_I2S + INMP441");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    apnea_count = 0;
    loop_count = 0;

    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n",
        sizeof(ei_classifier_inferencing_categories) /
        sizeof(ei_classifier_inferencing_categories[0]));

    ei_printf("\nStarting inference in 2 seconds...\n");
    delay(2000);

    if (!microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT)) {
        ei_printf("ERR: Failed to allocate audio buffer\r\n");
        return;
    }

    ei_printf("Recording...\n");
}

/**
 * @brief Main loop
 */
void loop()
{
    ei_printf("Loop: %d\n", ++loop_count);
    ei_printf("Starting recording...\n");

    if (!microphone_inference_record()) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    ei_printf("Recording done. Processing...\n");

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;

    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: run_classifier failed (%d)\n", r);
        return;
    }

    ei_printf("Predictions (DSP: %d ms, Class: %d ms):\n",
              result.timing.dsp, result.timing.classification);

    bool apnea_detected = false;

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("  %s: ", result.classification[ix].label);
        ei_printf_float(result.classification[ix].value);
        ei_printf("\n");

        if (strcmp(result.classification[ix].label, "ObstructiveApnea") == 0 &&
            result.classification[ix].value > APNEA_THRESHOLD) {
            apnea_detected = true;
        }
    }

    if (apnea_detected) {
        ei_printf(">>> ALARM: OBSTRUCTIVE APNEA DETECTED <<<\n");
        ei_printf("Apnea count: %d\n", ++apnea_count);
        digitalWrite(LED_PIN, HIGH);
    }
    else {
        digitalWrite(LED_PIN, LOW);
    }

    delay(1000);
}

/* Audio callbacks --------------------------------------------------------- */

static void audio_inference_callback(uint32_t n_bytes)
{
    for (int i = 0; i < n_bytes >> 1; i++) {
        inference.buffer[inference.buf_count++] = sampleBuffer[i];

        if (inference.buf_count >= inference.n_samples) {
            inference.buf_count = 0;
            inference.buf_ready = 1;
        }
    }
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void)
{
    I2S.end();
    free(inference.buffer);
}

/* Sampling task ----------------------------------------------------------- */

static void capture_samples(void* arg)
{
    const int32_t i2s_bytes_to_read = (uint32_t)arg;
    delay(100);

    while (record_status) {
        size_t bytes_read = I2S.readBytes((char *)sampleBuffer, i2s_bytes_to_read);

        if (bytes_read > 0) {
            for (int x = 0; x < bytes_read / 2; x++) {
                sampleBuffer[x] = (int16_t)(sampleBuffer[x]) * 2;
            }

            if (record_status) {
                audio_inference_callback(bytes_read);
            }
            else {
                break;
            }
        }
        else {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        vTaskDelay(1);
    }

    vTaskDelete(NULL);
}

/* Microphone init --------------------------------------------------------- */

static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffer = (int16_t *)heap_caps_malloc(
        n_samples * sizeof(int16_t), MALLOC_CAP_SPIRAM);

    if (inference.buffer == NULL) {
        ei_printf("Failed to allocate PSRAM buffer!\n");
        return false;
    }

    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    I2S.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);

    if (!I2S.begin(I2S_MODE_STD, EI_CLASSIFIER_FREQUENCY,
                   I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
        ei_printf("Failed to initialize I2S!\n");
        free(inference.buffer);
        return false;
    }

    record_status = true;
    xTaskCreate(capture_samples, "CaptureSamples", 4096,
                (void*)sample_buffer_size, 10, NULL);

    return true;
}

static bool microphone_inference_record(void)
{
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while (inference.buf_ready == 0) {
        delay(10);
    }

    inference.buf_ready = 0;
    return true;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
