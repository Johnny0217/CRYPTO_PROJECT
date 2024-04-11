## A Project regarding with Crypto currency

### Exec_mode
1. longshort (mainly focus on)
2. timeseries

### Introduction
- 第一次使用时请修改congif.ini中的PROJECT_PATH到本地该项目地址
- 所有的测试因子都在factor_test文件夹中，里面的因子跑完不会保存结果，只会保存临时结果在内存中
- 归档的因子在factors中，已经测试通过可以使用
- 回测均使用矩阵化回测
- modules_alternative_ETHBTC_backtest.py & alternative_ETHBTC_backtest.ipynb只是ETH BTC的时序组合因子测试结果，暂时不建议使用
- 截面多空以及时序模式均在modules_select_coin_backtest.py文件中，带mp是参数遍历版本，out=1查看样本外表现
- 一旦有测试通过的因子，仓位，pnl会通过modules_factor_mgmt.py文件保存结果到optimization文件中，再通过notebook查看相关性以及组合结果，目前只是等权组合
- 图片均保存到factors文件夹中，可以在utils修改保存路径，或者不保存直接展示
- 该框架同样适用于CTA，但是不适用于高频数据低频使用，需要进行groupby的转换，同频率矩阵化回测可以使用该框架


### Work Flow
- factors: contains set down factors -> results saved in optimization
- factor_test: will not save any result, a flush terminal running
- modules_corr_combo: will save results to optimizations, like a factor storage, do analysis with current existing factors, like correlation, portfolio


### Universe
Intersection with coinbase & binance, total number reaches 78


### CTA
202206 - 202305 DD if can cover drawdown 
Convex optimize


