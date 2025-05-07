import logging
import logging.handlers

logger = logging.getLogger('babeldoc_gui')  # アプリ名
logger.setLevel(logging.INFO)

# SysLogHandlerでリモートrsyslogサーバを指定（UDPデフォルト、ポート514）
handler = logging.FileHandler('babeldoc_gui.log')
formatter = logging.Formatter('%(name)s: %(levelname)s %(message)s')  # 書式は自由
handler.setFormatter(formatter)
logger.addHandler(handler)

# ログ送信
# logger.info('リモートrsyslogへのテストメッセージ')
