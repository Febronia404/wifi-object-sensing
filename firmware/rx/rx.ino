#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"
#include <math.h>

#define WIFI_CHANNEL 11        // must match the transmitter
#define N_SUB 64

struct CsiRow {
  uint32_t ts;
  int8_t rssi;
  uint8_t ch;
  uint8_t amp[N_SUB];
};

QueueHandle_t q;
uint8_t txMac[6];
volatile bool locked = false;
volatile uint32_t dropped = 0;

// Called when an ESP-NOW packet arrives: remember the transmitter MAC once
void onRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (!locked && len == 4) {
    memcpy(txMac, info->src_addr, 6);
    locked = true;
  }
}

// Called for every CSI measurement
void csiCb(void *ctx, wifi_csi_info_t *info) {
  if (!locked || !info || !info->buf) return;
  if (memcmp(info->mac, txMac, 6) != 0) return;   // ignore any other source
  if (info->len < N_SUB * 2) return;

  CsiRow r;
  r.ts = millis();
  r.rssi = info->rx_ctrl.rssi;
  r.ch = info->rx_ctrl.channel;

  const int8_t *b = info->buf;                    // (imag, real) per subcarrier
  for (int i = 0; i < N_SUB; i++) {
    float im = b[2 * i];
    float re = b[2 * i + 1];
    int a = (int)(sqrtf(re * re + im * im) + 0.5f);
    r.amp[i] = a > 255 ? 255 : a;
  }
  if (xQueueSend(q, &r, 0) != pdTRUE) dropped++;
}

void setup() {
  Serial.begin(500000);
  delay(1000);

  q = xQueueCreate(40, sizeof(CsiRow));

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(200);

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  if (esp_now_init() != ESP_OK) {
    Serial.println("# ESP-NOW init FAILED");
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(onRecv);

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;          // LLTF only: 64 fixed subcarriers
  cfg.htltf_en = false;
  cfg.stbc_htltf2_en = false;
  cfg.ltf_merge_en = false;
  cfg.channel_filter_en = false;
  cfg.manu_scale = false;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(csiCb, NULL);
  esp_wifi_set_csi(true);

  // Header
  Serial.print("timestamp_ms,rssi,channel");
  for (int i = 0; i < N_SUB; i++) Serial.printf(",a%d", i);
  Serial.println();
  Serial.println("# RX ready, waiting for TX...");
}

void loop() {
  CsiRow r;
  static uint32_t lastStatus = 0;
  static char line[512];

  while (xQueueReceive(q, &r, pdMS_TO_TICKS(50)) == pdTRUE) {
    int n = snprintf(line, sizeof(line), "%lu,%d,%u",
                     (unsigned long)r.ts, r.rssi, r.ch);
    for (int i = 0; i < N_SUB; i++) {
      n += snprintf(line + n, sizeof(line) - n, ",%u", r.amp[i]);
    }
    line[n++] = '\n';
    Serial.write((uint8_t*)line, n);
  }

  if (millis() - lastStatus > 3000) {
    lastStatus = millis();
    Serial.printf("# locked=%d dropped=%lu\n", locked ? 1 : 0, (unsigned long)dropped);
  }
}
