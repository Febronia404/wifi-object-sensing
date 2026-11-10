#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"

#define WIFI_CHANNEL 11        // must match the receiver
#define SEND_INTERVAL_MS 20    // 20 ms = 50 packets/sec

uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct Packet {
  uint32_t seq;
};

uint32_t seq = 0;
uint32_t lastPrint = 0;
uint32_t sentInSec = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(200);

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init FAILED");
    while (true) delay(1000);
  }

  // OFDM rate so that CSI is produced
  esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_MCS0_LGI);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, broadcastAddr, 6);
  peer.channel = WIFI_CHANNEL;
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("add_peer FAILED");
    while (true) delay(1000);
  }

  uint8_t ch; wifi_second_chan_t sc;
  esp_wifi_get_channel(&ch, &sc);
  Serial.printf("TX ready. channel=%d\n", ch);
}

void loop() {
  Packet p = { seq++ };
  if (esp_now_send(broadcastAddr, (uint8_t*)&p, sizeof(p)) == ESP_OK) {
    sentInSec++;
  }

  if (millis() - lastPrint >= 1000) {
    Serial.printf("sent/sec = %u (total %u)\n", sentInSec, seq);
    sentInSec = 0;
    lastPrint = millis();
  }
  delay(SEND_INTERVAL_MS);
}
