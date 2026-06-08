Ferrous
MovingAverageBias（State machine，N = 30） + OvernightVsIntraday（State machine，N = 20） + VolumeMomentum（RAW，N = 50）

NonFerrous（cu al）
MovingAveragebias（State machine，N = 40） + MACD（State machine，{'fast_n': 24, 'slow_n': 52, 'signal_n': 18}）

NonFerrous(others)
DualMACrossover（State machine,{'fast_n': 20, 'slow_n': 60}）+ BuyingSellingPressure（State machine，N = 30）

Energy
movingaveragebais（State machine，N = 40） + CMF（RAW，N=40）

Precious
DualMACrossover(20, 40) + DualMACrossover(20, 60) + BuyingSellingPressure(40)

Agriculture(油脂油料)
CMF(state machine 50)

Agriculture(软商品)
DonchianChannel_{State machine  N: 50}  
