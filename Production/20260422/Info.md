Ferrous
TrendMomentum_MovingAverageBias（State machine，N = 30 
CrossSectional_OvernightVsIntraday（State machine，N = 20）
Volume_VolumeMomentum（RAW，N = 50）

NonFerrous（cu al）
TrendMomentum_MovingAverageBias（State machine，N = 40）
TrendMomentum_MACD（State machine，{'fast_n': 24, 'slow_n': 52, 'signal_n': 18}）

NonFerrous(others)
TrendMomentum_DualMACrossover（State machine,{'fast_n': 20, 'slow_n': 60}）Microstructure_BuyingSellingPressure（State machine，N = 30）

Energy
TrendMomentum_MovingAverageBias（State machine，N = 40）
Volume_CMF（RAW，N=40）

Precious
TrendMomentum_DualMACrossover(State machine,20, 40)
TrendMomentum_DualMACrossover(State machine,20, 60)
Microstructure_BuyingSellingPressure(State machine,40)

Agriculture(油脂油料)
Volume_CMF(state machine 50)

Agriculture(软商品)
TrendMomentum_DonchianChannel{State machine  N: 50}  
