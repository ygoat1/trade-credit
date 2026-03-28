import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（解决matplotlib中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def input_game_data():
    """
    数据输入模块：支持手动输入或从文件读取博弈参数
    返回参数：N(厂商数), a, b(反需求函数参数), c_list(各厂商成本), K(资源上限)
    """
    print("="*50)
    print("广义纳什均衡博弈模型 - 数据输入")
    print("="*50)
    
    # 选择输入方式
    input_mode = input("请选择输入方式（1-手动输入，2-文件读取）：")
    while input_mode not in ["1", "2"]:
        input_mode = input("输入错误！请选择1或2：")
    
    if input_mode == "1":
        # 手动输入参数
        print("\n【手动输入参数】")
        while True:
            try:
                N = int(input("1. 厂商数量 N："))
                if N >= 2:
                    break
                print("厂商数量至少为2！")
            except ValueError:
                print("请输入整数！")
        
        while True:
            try:
                a = float(input("2. 反需求函数参数 a（>0）："))
                b = float(input("3. 反需求函数参数 b（>0）："))
                if a > 0 and b > 0:
                    break
                print("a和b必须大于0！")
            except ValueError:
                print("请输入数字！")
        
        c_list = []
        print(f"4. 各厂商边际成本 c_i（<{a}）：")
        for i in range(N):
            while True:
                try:
                    c = float(input(f"   厂商{i+1}的c_{i+1}："))
                    if c < a:
                        c_list.append(c)
                        break
                    print(f"c_{i+1}必须小于{a}！")
                except ValueError:
                    print("请输入数字！")
        
        while True:
            try:
                K = float(input("5. 共享资源上限 K（>0）："))
                if K > 0:
                    break
                print("K必须大于0！")
            except ValueError:
                print("请输入数字！")
    
    else:
        print("\n【文件读取参数】")
        file_path = input("请输入文件路径：")
        while True:
            try:
                df = pd.read_excel(file_path)

                # 读取基础参数
                N = int(df["N"].iloc[0])
                a = float(df["a"].iloc[0])
                b = float(df["b"].iloc[0])
                K = float(df["K"].iloc[0])
                
                # 读取成本参数
                c_list = []
                for i in range(N):
                    col_name = f"c{i+1}"
                    if col_name not in df.columns:
                        raise ValueError(f"缺少厂商{i+1}的成本列 {col_name}")
                    c = float(df[col_name].iloc[0])
                    c_list.append(c)
                
                # 验证参数合法性
                if N < 2:
                    raise ValueError("厂商数量N必须≥2")
                if a <= 0 or b <= 0 or K <= 0:
                    raise ValueError("a, b, K必须>0")
                if any(c >= a for c in c_list):
                    raise ValueError("所有c_i必须<a")
                
                print("文件读取成功！")
                break
            except FileNotFoundError:
                file_path = input("文件不存在！请重新输入路径：")
            except Exception as e:
                file_path = input(f"读取错误：{str(e)}！请重新输入路径：")
    
    # 显示输入参数确认
    print("\n" + "="*50)
    print("输入参数确认")
    print("="*50)
    print(f"厂商数量：{N}")
    print(f"反需求函数：p(Q) = {a} - {b}*Q")
    print(f"各厂商成本：c = {c_list}")
    print(f"共享资源上限：K = {K}")
    print("="*50)
    
    return N, a, b, c_list, K

def build_F_function(N, a, b, c_list):
    """
    构建变分不等式的映射函数 F(x)
    F(x) = 负的利润梯度（用于VI最优性条件）
    """
    def F(x):
        """
        x: 产量向量 [q1, q2, ..., qN]
        返回：F(x)向量
        """
        Q = np.sum(x)  # 总产量
        fx = np.zeros(N)
        for i in range(N):
            # 利润梯度：∂π_i/∂q_i = a - bQ - bq_i - c_i
            # F(x)_i = -∂π_i/∂q_i = bq_i + bQ + c_list[i] - a
            fx[i] = b * x[i] + b * Q + c_list[i] - a
        return fx
    return F

def proj_X(y, K):
    """
    投影操作：将点y投影到可行集X
    X = {x ≥ 0, sum(x) ≤ K}
    使用cvxpy求解二次规划实现投影
    """
    N = len(y)
    q = cp.Variable(N)
    # 目标函数：min ||q - y||²（欧氏投影）
    objective = cp.Minimize(cp.sum_squares(q - y))
    # 约束条件：非负产量 + 资源约束
    constraints = [q >= 0, cp.sum(q) <= K]
    # 构建并求解问题
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    return q.value

def solve_gne_via_vi(N, a, b, c_list, K):
    """
    使用变分不等式+投影算法求解广义纳什均衡
    返回：GNE解q_star, 迭代次数, 拉格朗日乘子lambda, 收敛历史
    """
    # 1. 构建F函数
    F = build_F_function(N, a, b, c_list)
    
    # 2. 投影算法参数设置
    np.random.seed(42)  # 固定随机种子保证可复现
    x_init = np.random.uniform(0, K/N, N)  # 初始点（均匀分布在可行集内）
    alpha = 0.005  # 步长（根据Lipschitz常数调整）
    max_iter = 10000  # 最大迭代次数
    tol = 1e-6  # 收敛阈值（两次迭代差值的2范数）
    
    # 3. 迭代求解（记录收敛过程）
    x = x_init.copy()
    iter_count = 0
    convergence_history = []  # 记录每次迭代的误差
    for k in range(max_iter):
        iter_count += 1
        # 计算F(x)
        fx = F(x)
        # 迭代更新：x^{k+1} = proj_X(x^k - alpha*F(x^k))
        x_new = proj_X(x - alpha * fx, K)
        # 记录误差
        error = np.linalg.norm(x_new - x)
        convergence_history.append(error)
        # 收敛判断
        if error < tol:
            print(f"\n迭代收敛！共迭代{iter_count}次")
            break
        x = x_new
    else:
        print(f"\n警告：已达到最大迭代次数{max_iter}，可能未完全收敛")
    
    # 4. 计算拉格朗日乘子（资源约束的影子价格）
    q_star = x
    Q_star = np.sum(q_star)
    lambda_list = F(q_star)  # 约束紧时所有厂商的λ应相等
    lambda_avg = np.mean(lambda_list)
    
    return q_star, iter_count, lambda_avg, Q_star, convergence_history

def verify_gne(q_star, N, a, b, c_list, K):
    """
    验证GNE解的有效性
    """
    print("\n" + "="*50)
    print("GNE解验证")
    print("="*50)
    
    # 1. 可行性验证
    Q_star = np.sum(q_star)
    feasible = all(q >= -1e-6 for q in q_star) and (Q_star <= K + 1e-6)
    print(f"1. 可行性验证：{'通过' if feasible else '失败'}")
    print(f"   各厂商产量非负：{all(q >= -1e-6 for q in q_star)}")
    print(f"   总产量 ≤ K：{Q_star:.4f} ≤ {K} {'(约束紧)' if abs(Q_star-K) < 1e-3 else ''}")
    
    # 2. 一阶条件验证（边际收益一致性）
    marginal_revenue = []
    for i in range(N):
        mr = a - b * Q_star - b * q_star[i] - c_list[i]
        marginal_revenue.append(mr)
    mr_max_diff = np.max(marginal_revenue) - np.min(marginal_revenue)
    print(f"\n2. 一阶条件验证：{'通过' if mr_max_diff < 1e-3 else '失败'}")
    print(f"   各厂商边际收益（应近似相等）：{[f'{mr:.4f}' for mr in marginal_revenue]}")
    print(f"   边际收益最大差值：{mr_max_diff:.6f}")
    
    return feasible and (mr_max_diff < 1e-3)

def calculate_evaluation_metrics(q_star, N, a, b, c_list, K):
    """
    计算扩展的评估指标
    """
    Q_star = np.sum(q_star)
    p_star = a - b * Q_star  # 均衡价格
    
    # 基础指标
    metrics = {
        "均衡价格": p_star,
        "总产量": Q_star,
        "资源利用率": Q_star / K * 100,
        "市场总收益": p_star * Q_star,
        "行业总利润": 0,
        "厂商利润": [],
        "厂商市场份额": [],
        "成本利润率": []
    }
    
    # 厂商级指标
    for i in range(N):
        profit = (p_star - c_list[i]) * q_star[i]
        market_share = q_star[i] / Q_star * 100 if Q_star > 0 else 0
        cost_margin = (profit / (c_list[i] * q_star[i])) * 100 if (c_list[i] * q_star[i]) > 0 else 0
        
        metrics["厂商利润"].append(profit)
        metrics["厂商市场份额"].append(market_share)
        metrics["成本利润率"].append(cost_margin)
        metrics["行业总利润"] += profit
    
    # 消费者剩余（CS = ∫0^Q* (a - bQ) dQ - p*Q* = 0.5*b*Q*^2）
    metrics["消费者剩余"] = 0.5 * b * Q_star **2
    # 社会总福利（消费者剩余 + 行业总利润）
    metrics["社会总福利"] = metrics["消费者剩余"] + metrics["行业总利润"]
    
    return metrics

def plot_results(q_star, metrics, convergence_history, N, a, b, c_list, K):
    """
    可视化结果：生成多子图展示关键信息（修复气泡大小报错）
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('广义纳什均衡博弈模型分析结果', fontsize=16, fontweight='bold')
    
    # 子图1：厂商产量分布（饼图）
    ax1 = plt.subplot(2, 3, 1)
    labels = [f'厂商{i+1}' for i in range(N)]
    sizes = q_star
    colors = plt.cm.Set3(np.linspace(0, 1, N))
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                    colors=colors, startangle=90)
    ax1.set_title('厂商产量分布', fontsize=12, fontweight='bold')
    
    # 子图2：厂商利润对比（柱状图）
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(N)
    ax2.bar(x, metrics["厂商利润"], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('厂商编号')
    ax2.set_ylabel('利润')
    ax2.set_title('各厂商均衡利润', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'厂商{i+1}' for i in range(N)])
    # 添加数值标签
    for i, v in enumerate(metrics["厂商利润"]):
        ax2.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    # 子图3：收敛过程（折线图）
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(range(1, len(convergence_history)+1), convergence_history, 
            color='darkblue', linewidth=2, marker='.', markersize=4)
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('迭代误差（2范数）')
    ax3.set_title('算法收敛过程', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')  # 对数刻度更清晰
    ax3.grid(True, alpha=0.3)
    
    # 子图4：成本vs产量vs利润（散点图）
    ax4 = plt.subplot(2, 3, 4)
    # 修复1：处理利润为0/负数的情况，保证气泡大小为正
    profit_values = np.array(metrics["厂商利润"])
    # 替换负数为极小正数，避免size为负
    profit_positive = np.where(profit_values <= 0, 0.1, profit_values)
    # 修复2：归一化利润值，避免气泡过大/过小（缩放至50-500区间）
    if np.max(profit_positive) > np.min(profit_positive):
        profit_scaled = 50 + (profit_positive - np.min(profit_positive)) / (np.max(profit_positive) - np.min(profit_positive)) * 450
    else:
        profit_scaled = np.ones(N) * 200  # 利润全部相同时用固定大小
    
    scatter = ax4.scatter(c_list, q_star, s=profit_scaled,  # 使用修复后的size数组
                        c=range(N), cmap='viridis', alpha=0.7, edgecolors='black')
    ax4.set_xlabel('边际成本')
    ax4.set_ylabel('均衡产量')
    ax4.set_title('成本-产量-利润关系\n（气泡大小=利润）', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    # 添加厂商标签
    for i in range(N):
        ax4.annotate(f'厂商{i+1}', (c_list[i], q_star[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 子图5：市场份额vs成本利润率（散点图）
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(metrics["厂商市场份额"], metrics["成本利润率"], 
                c=colors, s=100, alpha=0.8, edgecolors='black')
    ax5.set_xlabel('市场份额（%）')
    ax5.set_ylabel('成本利润率（%）')
    ax5.set_title('市场份额vs成本利润率', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    # 添加厂商标签
    for i in range(N):
        ax5.annotate(f'厂商{i+1}', (metrics["厂商市场份额"][i], metrics["成本利润率"][i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 子图6：关键宏观指标（柱状图）
    ax6 = plt.subplot(2, 3, 6)
    macro_metrics = ['行业总利润', '消费者剩余', '社会总福利']
    macro_values = [metrics["行业总利润"], metrics["消费者剩余"], metrics["社会总福利"]]
    colors_macro = ['#ff7f0e', '#2ca02c', '#d62728']
    ax6.bar(macro_metrics, macro_values, color=colors_macro, edgecolor='black', alpha=0.8)
    ax6.set_ylabel('金额')
    ax6.set_title('宏观经济指标', fontsize=12, fontweight='bold')
    # 添加数值标签
    for i, v in enumerate(macro_values):
        ax6.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('GNE博弈模型分析结果.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n 可视化图表已生成并保存为：GNE博弈模型分析结果.png")

def output_results(q_star, lambda_avg, Q_star, N, a, b, c_list, K, metrics):
    """
    输出最终结果（包含扩展评估指标）
    """
    print("\n" + "="*60)
    print("广义纳什均衡（GNE）求解结果与扩展评估")
    print("="*60)
    print(f"【基础参数】")
    print(f"模型类型：带耦合约束的古诺博弈（共享资源多厂商生产）")
    print(f"反需求函数：p(Q) = {a} - {b}*Q")
    print(f"厂商数量：{N} | 资源上限：K = {K}")
    print(f"各厂商成本：c = {[f'{c:.4f}' for c in c_list]}")
    
    print(f"\n【均衡核心结果】")
    for i in range(N):
        print(f"厂商{i+1}：产量 q* = {q_star[i]:.4f} | 利润 π* = {metrics['厂商利润'][i]:.4f}")
    print(f"均衡价格：p* = {metrics['均衡价格']:.4f}")
    print(f"总产量：Q* = {Q_star:.4f} | 资源利用率：{metrics['资源利用率']:.2f}%")
    print(f"资源约束影子价格（λ）：{lambda_avg:.4f}")
    
    print(f"\n【扩展评估指标】")
    print(f"行业总利润：{metrics['行业总利润']:.4f}")
    print(f"消费者剩余：{metrics['消费者剩余']:.4f}")
    print(f"社会总福利：{metrics['社会总福利']:.4f}")
    print(f"市场总收益：{metrics['市场总收益']:.4f}")
    
    print(f"\n【厂商详细指标】")
    for i in range(N):
        print(f"厂商{i+1}：市场份额 = {metrics['厂商市场份额'][i]:.2f}% | 成本利润率 = {metrics['成本利润率'][i]:.2f}%")
    print("="*60)

def main():
    """
    主函数：完整流程调度（包含评估和可视化）
    """
    print("="*60)
    print("          广义纳什均衡博弈模型求解（变分不等式+投影算法）")
    print("="*60)
    
    # 1. 数据输入
    N, a, b, c_list, K = input_game_data()
    
    # 2. 求解GNE（记录收敛历史）
    print("\n开始求解广义纳什均衡...")
    q_star, iter_count, lambda_avg, Q_star, convergence_history = solve_gne_via_vi(N, a, b, c_list, K)
    
    # 3. 结果验证
    verify_success = verify_gne(q_star, N, a, b, c_list, K)
    
    # 4. 计算扩展评估指标
    metrics = calculate_evaluation_metrics(q_star, N, a, b, c_list, K)
    
    # 5. 结果输出（含扩展指标）
    output_results(q_star, lambda_avg, Q_star, N, a, b, c_list, K, metrics)
    
    # 6. 可视化结果
    print("\n生成可视化分析图表...")
    plot_results(q_star, metrics, convergence_history, N, a, b, c_list, K)
    
    # 7. 结果总结
    print("\n" + "="*60)
    print("模型分析总结")
    print("="*60)
    print(f" 求解状态：{'验证通过' if verify_success else '验证失败'}")
    print(f" 核心结论：")
    print(f"   - 成本最低的厂商（厂商{np.argmin(c_list)+1}）获得最高市场份额：{metrics['厂商市场份额'][np.argmin(c_list)]:.2f}%")
    print(f"   - 资源利用率：{metrics['资源利用率']:.2f}% {'（资源未充分利用）' if metrics['资源利用率'] < 99 else '（资源接近耗尽）'}")
    print(f"   - 社会总福利：{metrics['社会总福利']:.4f}（消费者剩余占比：{metrics['消费者剩余']/metrics['社会总福利']*100:.2f}%）")
    print("="*60)

if __name__ == "__main__":
    main()
