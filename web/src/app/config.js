/**
 * 全局配置：常量、阈值、storage keys、默认 URL
 */

export const APP = {
  NAME: 'PetMind',
  VERSION: '2.0.0',
};

/** 与底层固件一致：50Hz 采样率 */
export const SAMPLING_RATE = 50;

export const BLE_BUFFER = {
  MAX: 5000,
  HARD_MAX: 5200,
};

export const VITAL = {
  HISTORY_LEN: 200,
  HEART_INIT: 70,
  RESP_INIT: 18,
  HEART_DELTA_BPM: 5,
  HEART_RATE_MIN: 48,
  HEART_RATE_MAX: 180,
  RESP_RATE_MIN: 6,
  RESP_RATE_MAX: 30,
  FILTER_WINDOW: 10,
  HEART_RATE_SMOOTHING: 30,
};

export const CHART = {
  MIN_INTERVAL_MS: 100,
  ADAPTIVE_THRESHOLD: 30,
  ADAPTIVE_WINDOW: 50,
};

export const URLS = {
  AGENT_DEFAULT: 'http://127.0.0.1:8000',
  INTEGRATION_DEFAULT: 'http://127.0.0.1:8000/integration/ingest',
};

export const STORAGE_KEYS = {
  AGENT_ENDPOINT: 'agentEndpoint',
  AGENT_API_KEY: 'agentApiKey',
  AGENT_MODEL: 'agentModel',
  CHAT_ANIMAL_ID: 'chatAnimalId',
  CHAT_HISTORY: 'petmindChatHistory',
  AZURE_CONFIG: 'azureConfig',
  BLE_UPLOAD_URL: 'bleUploadUrl',
  BLE_UPLOAD_INTERVAL: 'bleUploadInterval',
  BLE_ANIMAL_ID: 'bleAnimalId',
  BLE_DEVICE_ID: 'bleDeviceId',
  BLE_SAVED_DEVICE: 'bleSavedDevice',
  INTEGRATION_BASE: 'integrationIngestBase',
  CUSTOM_PROMPTS: 'petmind.customPrompts',
  RAG_DATABASE: 'petmind.ragDatabase',
};

export const DEFAULTS = {
  AGENT_API_KEY: 'sk-pethealthai-default-key-2026',
  AGENT_MODEL: 'agent-multi-turn',
};
