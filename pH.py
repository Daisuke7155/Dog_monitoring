import time
import spidev
from gpiozero import LED

# MCP3008の設定
SPI_CHANNEL = 0
SPI_SPEED = 1000000

# pHメーター設定
OFFSET = 0.00
SAMPLING_INTERVAL = 0.02  # 20ms
PRINT_INTERVAL = 0.8  # 800ms
ARRAY_LENGTH = 40

# MCP3008の読み取り関数
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# 平均値計算関数
def average_array(arr):
    if len(arr) <= 0:
        print("Error number for the array to averaging!\n")
        return 0
    if len(arr) < 5:
        return sum(arr) / len(arr)
    else:
        min_val = min(arr)
        max_val = max(arr)
        total = sum(arr) - min_val - max_val
        return total / (len(arr) - 2)

# ミリボルトからpH値を計算する関数
def millivolts_to_ph(mv):
    if mv >= 0:
        ph = 7.0 - (mv / 59.16)
    else:
        ph = 7.0 - (mv / 59.16)
    return ph

# 初期化
spi = spidev.SpiDev()
spi.open(0, SPI_CHANNEL)
spi.max_speed_hz = SPI_SPEED

led = LED(13)

pH_array = []
sampling_time = time.time()
print_time = time.time()

try:
    while True:
        if time.time() - sampling_time > SAMPLING_INTERVAL:
            pH_array.append(read_adc(0))
            if len(pH_array) > ARRAY_LENGTH:
                pH_array.pop(0)
            voltage = average_array(pH_array) * 5 / 1023  # MCP3008の出力を電圧に変換
            millivolts = voltage * 1000  # ボルトをミリボルトに変換
            pH_value = millivolts_to_ph(millivolts)
            sampling_time = time.time()
        
        if time.time() - print_time > PRINT_INTERVAL:
            print(f"Millivolts: {millivolts:.2f} mV    pH value: {pH_value:.2f}")
            led.toggle()
            print_time = time.time()

except KeyboardInterrupt:
    spi.close()
