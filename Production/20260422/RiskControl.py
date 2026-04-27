import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def convert_simple_to_compound(file_path, output_path, start_date=None, target_volatility=0.16, divisor=100.0):
    """
    将单利PnL转换为复利净值。
    
    参数:
    file_path: 输入的单利PnL csv文件路径
    output_path: 输出的复合净值 csv文件路径
    start_date: 开始计算的日期，格式为字符串 'YYYYMMDD'，如 '20150101'。如果为None，则从第一天开始。
    target_volatility: 目标的年化波动率参数，默认 0.16（即 16%）。若设为 0.20，日收益会扩大 20%/16% 倍。
    divisor: 如果PnL是百分比（如 4.06 代表 4.06%），则使用 100。
             如果PnL已经是小数形式（如 0.0406 代表 4.06%），则使用 1。
    """
    # 读取CSV文件
    # 假设第一列为日期，第二列为PnL
    df = pd.read_csv(file_path, index_col=0)
    
    # 根据 start_date 过滤数据
    if start_date is not None:
        # 将 index 转换为相同类型（字符串）以进行安全比较
        df = df[df.index.astype(str) >= str(start_date)]
        if df.empty:
            print("警告: 给定的起始日期之后没有数据，请检查起止日期。")
            return
    
    # 提取PnL列名的名称
    pnl_col = df.columns[0]
    
    # 按照设定的年化波动率计算杠杆比例 (默认基准为16%)
    volatility_scale = target_volatility / 0.16
    
    # 计算每日收益率 (单利形式下的每日收益率 = 每日PnL / 初始本金) 并加入杠杆
    daily_return = (df[pnl_col] / divisor) * volatility_scale
    
    # 初始化变量
    net_values = []
    positions = []
    compound_pnls = []
    drawdowns = []
    max_drawdowns = []
    
    current_net_value = 1.0
    max_net_value = 1.0
    current_position = 1.0
    running_max_dd = 0.0
    
    for i, ret in enumerate(daily_return):
        # 记录当天的仓位
        positions.append(current_position)
        
        # 计算当天的实际收益 (叠加仓位系数)
        actual_ret = ret * current_position
        compound_pnls.append(actual_ret)
        
        # 更新净值
        current_net_value = current_net_value * (1 + actual_ret)
        net_values.append(current_net_value)
        
        # 更新历史最高净值
        if current_net_value > max_net_value:
            max_net_value = current_net_value
            
        # 计算当前回撤
        drawdown = 1.0 - (current_net_value / max_net_value)
        drawdowns.append(drawdown)
        
        # 记录历史最大回撤
        if drawdown > running_max_dd:
            running_max_dd = drawdown
        max_drawdowns.append(running_max_dd)
        
        # 根据回撤更新下一次的仓位系数
        if drawdown <= 0.05:
            current_position = 1.0
        elif drawdown <= 0.10:
            current_position = 0.8
        elif drawdown <= 0.15:
            current_position = 0.5
        else:
            current_position = 0.3
            
    # 组装结果
    result_df = pd.DataFrame({
        'Simple_PnL': df[pnl_col],
        'Position_Factor': positions,
        'Compound_PnL_Rate': compound_pnls,
        'Compound_Net_Value': net_values,
        'Drawdown': drawdowns,
        'Max_Drawdown': max_drawdowns
    }, index=df.index)
    
    # 将结果写出
    result_df.to_csv(output_path)
    print(f"转换成功！结果已保存至: {output_path}")

    # 计算 Sharpe Ratio (假设年化 252 个交易日)
    mean_return = np.mean(compound_pnls)
    std_return = np.std(compound_pnls)
    sharpe_ratio = np.sqrt(252) * mean_return / std_return if std_return != 0 else 0
    
    # 拿到最大的 MDD
    mdd = max(max_drawdowns)
    
    # 绘制净值图
    plt.figure(figsize=(10, 6))
    x_data = range(len(result_df))
    plt.plot(x_data, result_df['Compound_Net_Value'], label='Compound Net Value')
    
    # 动态选取大约 8 个交易日作为横轴标签，避免拥挤重叠
    num_ticks = 8
    tick_positions = np.linspace(0, len(result_df) - 1, num_ticks, dtype=int)
    tick_labels = [str(result_df.index[i]) for i in tick_positions]
    
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(f"Compound Net Value\nSharpe Ratio: {sharpe_ratio:.2f} | MDD: {mdd:.2%}")
    plt.xlabel('Date')
    plt.ylabel('Net Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    pic_path = output_path.replace('.csv', '.png')
    plt.savefig(pic_path)
    print(f"净值图已保存至: {pic_path}")

if __name__ == "__main__":
    input_file = r"d:\CTA\Production\20260422\result\L3_Sector_Merge_All_norm_PnL.csv"
    output_file = r"d:\CTA\Production\20260422\result\L3_Sector_Merge_All_compound_net_value.csv"
    
    # 您可以在在此处设置 start_date='20100104' (示例) 或者保持为 None 
    # 如果设置了 start_date，脚本将从该日期及其之后的数据开始计算复利
    target_volatility = 0.16 # 可以按需修改为 0.20
    convert_simple_to_compound(input_file, output_file, start_date='20190101', target_volatility=0.2, divisor=100.0)
