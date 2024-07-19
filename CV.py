import time
from gpiozero import MCP3008

class DFRobot_EC:
    def __init__(self):
        self.temperature = 25.0  # 固定温度値

    def readEC(self, voltage, temperature):
        ecValue = voltage  # 例として簡単な変換式
        return ecValue

    def calibration(self, voltage, temperature):
        # キャリブレーションのロジックを追加
        pass

# MCP3008 の CH0 ピンに接続されたアナログセンサーを読み取る
adc = MCP3008(channel=1)
ec = DFRobot_EC()

def main():
    while True:
        # MCP3008 からの読み取りは0〜1の範囲の比率を返すため、3.3Vにスケールする
        voltage = adc.value * 5 * 1000  # ミリボルトに変換
        temperature = ec.temperature  # 固定温度値
        ecValue = ec.readEC(voltage, temperature)

        print(f"Temperature: {temperature:.1f} °C")
        print(f"EC: {ecValue:.2f} mS/m")

        # 必要に応じてキャリブレーションを実行
        ec.calibration(voltage, temperature)

        time.sleep(1)

if __name__ == "__main__":
    main()
