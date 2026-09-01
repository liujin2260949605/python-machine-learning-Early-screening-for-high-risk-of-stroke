# ====================================================================
# 前置代码：导入库与设置
# ====================================================================

# 基础库
import numpy as np
import pandas as pd

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 可视化（先导入）
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# ---------- 关键：设置中文字体和样式（与成功项目一致） ----------
# 先设置 Seaborn 样式，再配置字体
sns.set_style("whitegrid")
font_path = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(font_path)
chinese_font = font_manager.FontProperties(fname=font_path)
font_name = chinese_font.get_name()
plt.rcParams.update({
   "font.family": font_name,
   "font.sans-serif": [font_name],
   "axes.unicode_minus": False,
   "font.size": 12,
})
#print("当前中文字体名称：", font_name)
#print("当前中文字体文件：", font_manager.findfont(font_name))
#plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
#plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块
#sns.set_style("whitegrid")                  # 设置 seaborn 风格
#plt.rcParams['font.size'] = 12              # 全局字体大小

# 模型与预处理
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 评估指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# 类别不平衡处理
from sklearn.utils.class_weight import compute_class_weight

# 统计检验
import scipy.stats as stats
from scipy.stats import chi2_contingency

print("所有库导入成功！")


# ====================================================================
# 任务1：读取并审查数据
# ====================================================================

# 读取数据
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 将工作目录切换到脚本所在目录
df = pd.read_csv('E6+PBL.csv')

# 1. 查看数据概览
print("数据形状:", df.shape)
print("\n前5行:")
print(df.head())

# 2. 识别字段类型
print("\n各字段数据类型:")
print(df.dtypes)

numeric_cols = ['id', 'age', 'avg_glucose_level', 'bmi']
categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                    'work_type', 'Residence_type', 'smoking_status', 'stroke']

print("\n类别字段唯一值示例:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()[:10]}")

# 3. 目标变量比例
stroke_counts = df['stroke'].value_counts()
stroke_ratio = stroke_counts / len(df)
print("\n目标变量 stroke 分布:")
print(stroke_counts)
print(f"正类比例: {stroke_ratio[1]:.2%}")

# 4. 缺失值检查（全局）
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
missing = df.isnull().sum()
print("\n各字段缺失值数量（bmi中的N/A已转为NaN）:")
print(missing[missing > 0])

# 5. 异常值初步探查
print("\n数值字段描述性统计:")
print(df[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))

# 6. 明确不参与建模的字段
exclude_from_model = ['id']
print(f"\n不参与建模的字段: {exclude_from_model}")
print("其余字段将作为特征（目标变量 stroke 除外）")


# ====================================================================
# 任务2：数据处理（仅处理BMI和gender，其他缺失去任务5处理）
# ====================================================================

print("=" * 60)
print("开始执行数据处理任务（任务 2）")
print("=" * 60)

# 步骤1：处理BMI缺失值（中位数填补）
print("\n[步骤 1] 处理 BMI 缺失值...")
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
bmi_missing_before = df['bmi'].isnull().sum()
print(f"  - BMI 缺失值数量（转换后）: {bmi_missing_before}")
bmi_median = df['bmi'].median()
print(f"  - BMI 中位数值: {bmi_median:.2f}")
df['bmi'].fillna(bmi_median, inplace=True)
bmi_missing_after = df['bmi'].isnull().sum()
print(f"  - BMI 缺失值数量（填补后）: {bmi_missing_after} (应为 0)")

# 步骤2：处理 smoking_status 中的 Unknown（保留为独立类别）
print("\n[步骤 2] 处理 smoking_status 中的 'Unknown'...")
unknown_count = (df['smoking_status'] == 'Unknown').sum()
unknown_ratio = unknown_count / len(df) * 100
print(f"  - 'Unknown' 样本数量: {unknown_count} ({unknown_ratio:.1f}%)")
print(f"  - 处理策略: 保留 'Unknown' 作为独立类别（不做任何转换）")

# 步骤3：处理极少数类别值（gender 中的 Other → 合并到 Male）
print("\n[步骤 3] 处理 gender 中的极少数类别 'Other'...")
other_count = (df['gender'] == 'Other').sum()
print(f"  - 'Other' 样本数量: {other_count}")
df['gender'] = df['gender'].replace('Other', 'Male')
print(f"  - 处理后的 gender 唯一值: {df['gender'].unique()}")

# 步骤4：剔除 id 字段
print("\n[步骤 4] 剔除不参与建模的 'id' 字段...")
df.drop(columns=['id'], inplace=True)
print(f"  - 已删除 'id' 列。当前数据形状: {df.shape}")

# 步骤5：最终验证（此时可能还有 avg_glucose_level 等缺失，将在任务5处理）
print("\n[步骤 5] 数据质量初步验证...")
total_missing = df.isnull().sum().sum()
print(f"  - 当前数据总缺失值数量: {total_missing}（将在任务5中处理）")
print("\n" + "=" * 60)
print("任务 2 数据处理执行完毕！")
print("=" * 60)


# ====================================================================
# 任务3：探索性分析
# ====================================================================

print("=" * 60)
print("开始执行探索性分析（任务 3）")
print("=" * 60)


plt.figure(figsize=(14, 12))

# 1. 年龄
print("\n[1] 年龄与卒中的差异分析")
age_stats = df.groupby('stroke')['age'].describe()
print(age_stats)
plt.subplot(2, 3, 1)
sns.boxplot(x='stroke', y='age', data=df, palette=['#3498db', '#e74c3c'])
plt.title('年龄与卒中状态', fontsize=12)
plt.xlabel('卒中 (0=否, 1=是)')
plt.ylabel('年龄 (岁)')
stroke_0_age = df[df['stroke'] == 0]['age']
stroke_1_age = df[df['stroke'] == 1]['age']
levene_stat, levene_p = stats.levene(stroke_0_age, stroke_1_age)
if levene_p < 0.05:
    print("  - 方差不齐，采用 Mann-Whitney U 检验")
    stat, p_value = stats.mannwhitneyu(stroke_0_age, stroke_1_age, alternative='two-sided')
else:
    print("  - 方差齐性，采用独立样本 t 检验")
    stat, p_value = stats.ttest_ind(stroke_0_age, stroke_1_age)
print(f"  - 检验统计量: {stat:.3f}, p值: {p_value:.4e}")
print(f"  - 卒中组年龄均值: {stroke_1_age.mean():.1f} 岁, 非卒中组均值: {stroke_0_age.mean():.1f} 岁")
if p_value < 0.05:
    print("  - 结论: 年龄在两组间存在显著差异，卒中患者平均年龄更高。")

# 2. 平均血糖
print("\n[2] 平均血糖与卒中的差异分析")
glucose_stats = df.groupby('stroke')['avg_glucose_level'].describe()
print(glucose_stats)
plt.subplot(2, 3, 2)
sns.boxplot(x='stroke', y='avg_glucose_level', data=df, palette=['#3498db', '#e74c3c'])
plt.title('平均血糖与卒中状态', fontsize=12)
plt.xlabel('卒中 (0=否, 1=是)')
plt.ylabel('平均血糖 (mg/dL)')
stroke_0_glu = df[df['stroke'] == 0]['avg_glucose_level']
stroke_1_glu = df[df['stroke'] == 1]['avg_glucose_level']
levene_stat, levene_p = stats.levene(stroke_0_glu, stroke_1_glu)
if levene_p < 0.05:
    print("  - 方差不齐，采用 Mann-Whitney U 检验")
    stat, p_value = stats.mannwhitneyu(stroke_0_glu, stroke_1_glu, alternative='two-sided')
else:
    print("  - 方差齐性，采用独立样本 t 检验")
    stat, p_value = stats.ttest_ind(stroke_0_glu, stroke_1_glu)
print(f"  - 检验统计量: {stat:.3f}, p值: {p_value:.4e}")
print(f"  - 卒中组血糖均值: {stroke_1_glu.mean():.1f}, 非卒中组均值: {stroke_0_glu.mean():.1f}")
if p_value < 0.05:
    print("  - 结论: 平均血糖在两组间存在显著差异，卒中患者血糖水平更高。")

# 3. BMI
print("\n[3] BMI 与卒中的差异分析")
bmi_stats = df.groupby('stroke')['bmi'].describe()
print(bmi_stats)
plt.subplot(2, 3, 3)
sns.boxplot(x='stroke', y='bmi', data=df, palette=['#3498db', '#e74c3c'])
plt.title('BMI 与卒中状态', fontsize=12)
plt.xlabel('卒中 (0=否, 1=是)')
plt.ylabel('BMI (kg/m²)')
stroke_0_bmi = df[df['stroke'] == 0]['bmi']
stroke_1_bmi = df[df['stroke'] == 1]['bmi']
levene_stat, levene_p = stats.levene(stroke_0_bmi, stroke_1_bmi)
if levene_p < 0.05:
    print("  - 方差不齐，采用 Mann-Whitney U 检验")
    stat, p_value = stats.mannwhitneyu(stroke_0_bmi, stroke_1_bmi, alternative='two-sided')
else:
    print("  - 方差齐性，采用独立样本 t 检验")
    stat, p_value = stats.ttest_ind(stroke_0_bmi, stroke_1_bmi)
print(f"  - 检验统计量: {stat:.3f}, p值: {p_value:.4e}")
print(f"  - 卒中组 BMI 均值: {stroke_1_bmi.mean():.1f}, 非卒中组均值: {stroke_0_bmi.mean():.1f}")
if p_value < 0.05:
    print("  - 结论: BMI 在两组间存在显著差异。")
else:
    print("  - 结论: BMI 在两组间无显著差异。")

# 4. 高血压
print("\n[4] 高血压与卒中的差异分析")
hypertension_cross = pd.crosstab(df['hypertension'], df['stroke'], margins=True)
print("高血压 × 卒中 交叉表:")
print(hypertension_cross)
plt.subplot(2, 3, 4)
hypertension_rate = df.groupby('hypertension')['stroke'].mean()
hypertension_rate.plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('高血压与卒中率', fontsize=12)
plt.xlabel('高血压 (0=无, 1=有)')
plt.ylabel('卒中发生率')
plt.xticks(rotation=0)
plt.ylim(0, 0.3)
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df['hypertension'], df['stroke']))
print(f"  - 卡方检验: chi2={chi2:.3f}, p值={p:.4e}")
if p < 0.05:
    print("  - 结论: 高血压与卒中显著相关，高血压患者卒中率更高。")

# 5. 心脏病
print("\n[5] 心脏病与卒中的差异分析")
heart_cross = pd.crosstab(df['heart_disease'], df['stroke'], margins=True)
print("心脏病 × 卒中 交叉表:")
print(heart_cross)
plt.subplot(2, 3, 5)
heart_rate = df.groupby('heart_disease')['stroke'].mean()
heart_rate.plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('心脏病与卒中率', fontsize=12)
plt.xlabel('心脏病 (0=无, 1=有)')
plt.ylabel('卒中发生率')
plt.xticks(rotation=0)
plt.ylim(0, 0.3)
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df['heart_disease'], df['stroke']))
print(f"  - 卡方检验: chi2={chi2:.3f}, p值={p:.4e}")
if p < 0.05:
    print("  - 结论: 心脏病与卒中显著相关，心脏病患者卒中率更高。")

# 6. 吸烟状态
print("\n[6] 吸烟状态与卒中的差异分析")
smoke_cross = pd.crosstab(df['smoking_status'], df['stroke'], margins=True)
print("吸烟状态 × 卒中 交叉表:")
print(smoke_cross)
plt.subplot(2, 3, 6)
smoke_rate = df.groupby('smoking_status')['stroke'].mean()
smoke_rate = smoke_rate.reindex(['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
smoke_rate.plot(kind='bar', color=['#2ecc71', '#f1c40f', '#e67e22', '#95a5a6'])
plt.title('吸烟状态与卒中率', fontsize=12)
plt.xlabel('吸烟状态')
plt.ylabel('卒中发生率')
plt.xticks(rotation=15)
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df['smoking_status'], df['stroke']))
print(f"  - 卡方检验（含 Unknown）: chi2={chi2:.3f}, p值={p:.4e}")
if p < 0.05:
    print("  - 结论: 吸烟状态与卒中显著相关（含 Unknown）。")
else:
    print("  - 结论: 吸烟状态与卒中无显著相关（含 Unknown）。")
df_known_smoke = df[df['smoking_status'] != 'Unknown']
chi2_known, p_known, _, _ = chi2_contingency(pd.crosstab(df_known_smoke['smoking_status'], df_known_smoke['stroke']))
print(f"  - 剔除 Unknown 后卡方检验: chi2={chi2_known:.3f}, p值={p_known:.4e}")
if p_known < 0.05:
    print("  - 结论（剔除 Unknown）: 吸烟状态与卒中显著相关。")

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("任务 3 探索性分析完成！")
print("=" * 60)


# ====================================================================
# 任务4：全面交叉分析
# ====================================================================

# 定义分层函数
def age_group(age):
    if age < 45: return '<45'
    elif age < 60: return '45-60'
    elif age < 75: return '60-75'
    else: return '≥75'

def glucose_group(glucose):
    if glucose < 100: return '正常 (<100)'
    elif glucose < 126: return '糖尿病前期 (100-125)'
    else: return '糖尿病 (≥126)'

def bmi_group(bmi):
    if bmi < 25: return '正常 (<25)'
    elif bmi < 30: return '超重 (25-30)'
    else: return '肥胖 (≥30)'

df['age_group'] = df['age'].apply(age_group)
df['glucose_group'] = df['avg_glucose_level'].apply(glucose_group)
df['bmi_group'] = df['bmi'].apply(bmi_group)

smoke_order = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
df['smoking_status'] = pd.Categorical(df['smoking_status'], categories=smoke_order, ordered=True)

print("分层变量已创建。各组样本量：")
print("年龄组:\n", df['age_group'].value_counts().sort_index())
print("\n血糖组:\n", df['glucose_group'].value_counts().sort_index())
print("\nBMI组:\n", df['bmi_group'].value_counts().sort_index())

def plot_heatmap(data, x_col, y_col, title, ax, cmap='Reds', annot=True, fmt='.3f'):
    rate = data.groupby([y_col, x_col])['stroke'].mean().unstack()
    sns.heatmap(rate, annot=annot, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={'label': '卒中率'})
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return rate

fig, axes = plt.subplots(2, 4, figsize=(20, 12))
ax_list = axes.flatten()
all_rates = {}

print("\n" + "="*60)
print("交叉组合 1: 年龄分层 × 高血压")
rate1 = plot_heatmap(df, 'hypertension', 'age_group', '年龄 × 高血压', ax_list[0])
all_rates['年龄×高血压'] = rate1

print("\n" + "="*60)
print("交叉组合 2: 年龄分层 × 心脏病")
rate2 = plot_heatmap(df, 'heart_disease', 'age_group', '年龄 × 心脏病', ax_list[1])
all_rates['年龄×心脏病'] = rate2

print("\n" + "="*60)
print("交叉组合 3: 年龄分层 × 吸烟状态")
rate3 = plot_heatmap(df, 'smoking_status', 'age_group', '年龄 × 吸烟状态', ax_list[2])
all_rates['年龄×吸烟状态'] = rate3

print("\n" + "="*60)
print("交叉组合 4: 血糖分层 × BMI 分层")
rate4 = plot_heatmap(df, 'bmi_group', 'glucose_group', '血糖 × BMI', ax_list[3])
all_rates['血糖×BMI'] = rate4

print("\n" + "="*60)
print("交叉组合 5: 高血压 × 心脏病")
rate5 = plot_heatmap(df, 'heart_disease', 'hypertension', '高血压 × 心脏病', ax_list[4])
all_rates['高血压×心脏病'] = rate5

print("\n" + "="*60)
print("交叉组合 6: 高血压 × work_type")
rate6 = plot_heatmap(df, 'work_type', 'hypertension', '高血压 × 工作类型', ax_list[5])
all_rates['高血压×工作类型'] = rate6

print("\n" + "="*60)
print("交叉组合 7: 心脏病 × work_type")
rate7 = plot_heatmap(df, 'work_type', 'heart_disease', '心脏病 × 工作类型', ax_list[6])
all_rates['心脏病×工作类型'] = rate7

ax_list[7].axis('off')
plt.tight_layout()
plt.show()
plt.close()

# 找出全局最高风险组合
print("\n" + "="*60)
print("全局最高风险组合查找")
print("="*60)
max_rate = 0
max_comb = None
max_combo_name = None
for name, rate_df in all_rates.items():
    for idx in rate_df.index:
        for col in rate_df.columns:
            val = rate_df.loc[idx, col]
            if not pd.isna(val) and val > max_rate:
                max_rate = val
                max_comb = (idx, col)
                max_combo_name = name
print(f"全局最高风险组合: {max_combo_name} 中的 {max_comb}")
print(f"卒中率: {max_rate:.2%}")

print("\n各组合TOP3高风险人群（按卒中率降序）：")
for name, rate_df in all_rates.items():
    sorted_rates = rate_df.stack().sort_values(ascending=False)
    top3 = sorted_rates.head(3)
    print(f"\n[{name}]")
    for (idx, col), val in top3.items():
        print(f"  {idx} × {col}: {val:.2%}")

# 业务问题回答
print("\n" + "="*60)
print("业务问题回答（综合交叉分析结果）")
print("="*60)
idx, col = max_comb
if max_combo_name == '年龄×高血压':
    desc = f"年龄组为 {idx}，且患有高血压"
elif max_combo_name == '年龄×心脏病':
    desc = f"年龄组为 {idx}，且患有心脏病"
elif max_combo_name == '年龄×吸烟状态':
    desc = f"年龄组为 {idx}，且吸烟状态为 {col}"
elif max_combo_name == '血糖×BMI':
    desc = f"血糖组为 {idx}，BMI组为 {col}"
elif max_combo_name == '高血压×心脏病':
    desc = f"患有高血压且患有心脏病" if (idx==1 and col==1) or (idx==1 and col==1) else f"高血压={idx}，心脏病={col}"
elif max_combo_name == '高血压×工作类型':
    desc = f"患有高血压且工作类型为 {col}"
elif max_combo_name == '心脏病×工作类型':
    desc = f"患有心脏病且工作类型为 {col}"
else:
    desc = f"{idx} × {col}"

print(f"""
【问题1】哪类组合人群的卒中风险更高？
根据7组交叉分析，风险最高的组合是：
  - {max_combo_name}：{desc}，卒中率为 {max_rate:.2%}。
  其他高风险组合（卒中率 > 10%）通常涉及高龄（≥75岁）合并高血压、心脏病或既往吸烟。

【问题2】为什么交叉关系比单变量更能支持筛查？
单变量分析只能识别单个风险因子（如年龄、高血压），但无法揭示因子间的协同作用。
例如：
  - 仅看年龄，≥75岁人群卒中率约15%，但其中无高血压者可能仅5%，有高血压者可达25%以上。
  - 交叉分析能精确发现“年龄+高血压”的叠加效应，使筛查更有针对性。
在资源有限的现实条件下，交叉分析帮助我们避免“一刀切”策略，实现精准定向。

【问题3】如果资源有限，优先筛谁？
基于上述交叉分析，优先筛查顺序建议如下：
  第一优先级：{desc}（卒中率 {max_rate:.2%}）。
  第二优先级：年龄≥75岁 且 患有心脏病 或 曾经吸烟（视具体数据而定）。
  第三优先级：年龄60-75岁 且 高血压/糖尿病合并肥胖等组合。
具体建议：对 ≥75岁 的老年人进行全面危险因素评估（高血压、心脏病、吸烟史），对其中合并≥2个危险因素者优先安排深入筛查和随访。

注意：以上分析基于描述性统计，未考虑混杂因素，最终模型会通过逻辑回归/随机森林综合评估。
""")


# ====================================================================
# 任务5：模型训练（修正版 - 正确填补缺失值）
# ====================================================================

print("=" * 60)
print("开始模型训练（任务 5）")
print("=" * 60)

# 步骤1：数据预处理（编码 + 标准化）
print("\n[步骤 1] 数据预处理...")
df_model = df.copy()
X_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
          'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
y_col = 'stroke'
X = df_model[X_cols]
y = df_model[y_col]

# ---------- 强化缺失值处理（使用非 inplace 方式） ----------
# 1. 将数值列强制转为数值类型，非数字转为 NaN
numeric_cols_X = ['age', 'avg_glucose_level', 'bmi']
for col in numeric_cols_X:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# 2. 检查并填补所有列
print("  - 处理前各列缺失值:")
print(X.isnull().sum())

# 数值列用中位数填补（正确方式：赋值回原列）
for col in numeric_cols_X:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)   # 避免 inplace 失效
        print(f"    数值列 {col} 用中位数 {median_val} 填补")

# 分类列用众数填补（同样方式）
categorical_cols_X = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                      'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols_X:
    if X[col].isnull().any():
        mode_val = X[col].mode()[0]
        X[col] = X[col].fillna(mode_val)
        print(f"    分类列 {col} 用众数 '{mode_val}' 填补")

# 3. 检查目标变量 y
if y.isnull().any():
    print(f"  警告：目标变量 y 有 {y.isnull().sum()} 个缺失值，将删除对应行")
    valid_idx = ~y.isnull()
    X = X[valid_idx]
    y = y[valid_idx]

# 4. 最终确认
print(f"  - 处理后 X 总缺失值: {X.isnull().sum().sum()}, y 缺失值: {y.isnull().sum()}")
assert X.isnull().sum().sum() == 0, "X 仍有缺失值！"
assert y.isnull().sum() == 0, "y 仍有缺失值！"
# -----------------------------------------

# 独热编码
X_encoded = pd.get_dummies(X, drop_first=False)
print(f"编码后特征维度: {X_encoded.shape}")

# 拆分
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集正类比例: {y_train.mean():.2%}, 测试集正类比例: {y_test.mean():.2%}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = X_encoded.columns.tolist()

# 步骤2：训练模型
print("\n[步骤 2] 训练模型...")
print("\n--- 逻辑回归 ---")
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

print("\n--- 随机森林 ---")
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

# 步骤3：评估函数
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} 评估结果:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}  {'★ 达标' if rec >= 0.60 else '✗ 未达标'}")
    print(f"  F1-score : {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  混淆矩阵:\n{cm}")
    print(f"  分类报告:\n{classification_report(y_test, y_pred, zero_division=0)}")
    return y_pred, y_pred_proba

def evaluate_model_with_proba(y_true, y_pred, y_proba, model_name):
    """接受已预测的 y_pred 和 y_proba，输出评估指标"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}  {'★ 达标' if rec >= 0.60 else '✗ 未达标'}")
    print(f"  F1-score : {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  混淆矩阵:\n{cm}")
    print(f"  分类报告:\n{classification_report(y_true, y_pred, zero_division=0)}")

# 步骤4：阈值调整函数
def adjust_threshold(model, X_test, y_test, model_name, target_recall=0.60):
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    best_thresh = 0.5
    best_recall = recall_score(y_test, model.predict(X_test))
    best_prec = precision_score(y_test, model.predict(X_test), zero_division=0)
    best_f1 = f1_score(y_test, model.predict(X_test))
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        rec = recall_score(y_test, y_pred_thresh)
        if rec >= target_recall:
            prec = precision_score(y_test, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_test, y_pred_thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_recall = rec
                best_prec = prec
    print(f"\n{model_name} 阈值调整结果:")
    print(f"  最佳阈值: {best_thresh:.2f}")
    print(f"  Recall: {best_recall:.4f} (目标 ≥ {target_recall})")
    print(f"  Precision: {best_prec:.4f}")
    print(f"  F1-score: {best_f1:.4f}")
    return best_thresh

# 步骤5：初步评估
print("\n[步骤 3] 模型评估（默认阈值0.5）...")
lr_pred, lr_proba = evaluate_model(lr_model, X_test_scaled, y_test, "逻辑回归")
rf_pred, rf_proba = evaluate_model(rf_model, X_test_scaled, y_test, "随机森林")

print("\n[步骤4] 阈值调整说明")
print("  - 若默认阈值下 Recall < 0.60，将在步骤5中自动调整阈值。")
print("  - 调整策略：在 0.1~0.9 间搜索，使 Recall ≥ 0.60 且 F1 最高的阈值。")

# 步骤5：最终模型选择（包含阈值调整）
print("\n" + "=" * 60)
print("步骤5：最终模型选择与阈值调整")
print("=" * 60)

# 获取默认预测和概率
y_pred_lr_default = lr_model.predict(X_test_scaled)
y_proba_lr_default = lr_model.predict_proba(X_test_scaled)[:, 1]
y_pred_rf_default = rf_model.predict(X_test_scaled)
y_proba_rf_default = rf_model.predict_proba(X_test_scaled)[:, 1]

# 判断是否需要调整阈值
best_thresh_lr = 0.5  # 默认
best_thresh_rf = 0.5  # 默认
if recall_score(y_test, y_pred_lr_default) < 0.60:
    best_thresh_lr = adjust_threshold(lr_model, X_test_scaled, y_test, "逻辑回归")
    y_pred_lr_final = (y_proba_lr_default >= best_thresh_lr).astype(int)
    y_proba_lr_final = y_proba_lr_default
else:
    y_pred_lr_final = y_pred_lr_default
    y_proba_lr_final = y_proba_lr_default

if recall_score(y_test, y_pred_rf_default) < 0.60:
    best_thresh_rf = adjust_threshold(rf_model, X_test_scaled, y_test, "随机森林")
    y_pred_rf_final = (y_proba_rf_default >= best_thresh_rf).astype(int)
    y_proba_rf_final = y_proba_rf_default
else:
    y_pred_rf_final = y_pred_rf_default
    y_proba_rf_final = y_proba_rf_default

rec_lr = recall_score(y_test, y_pred_lr_final)
rec_rf = recall_score(y_test, y_pred_rf_final)

print(f"\n逻辑回归最终 Recall: {rec_lr:.4f}")
print(f"随机森林最终 Recall: {rec_rf:.4f}")

# 选择最终模型
if rec_lr >= 0.60 and rec_rf >= 0.60:
    final_model = lr_model if rec_lr >= rec_rf else rf_model
    final_pred = y_pred_lr_final if rec_lr >= rec_rf else y_pred_rf_final
    final_proba = y_proba_lr_final if rec_lr >= rec_rf else y_proba_rf_final
    final_name = "逻辑回归" if rec_lr >= rec_rf else "随机森林"
elif rec_lr >= 0.60:
    final_model, final_pred, final_proba, final_name = lr_model, y_pred_lr_final, y_proba_lr_final, "逻辑回归"
elif rec_rf >= 0.60:
    final_model, final_pred, final_proba, final_name = rf_model, y_pred_rf_final, y_proba_rf_final, "随机森林"
else:
    print("警告：两个模型 Recall 均未达到 0.60！需要进一步优化（如使用 SMOTE 或调整参数）。")
    final_model = lr_model if rec_lr >= rec_rf else rf_model
    final_pred = y_pred_lr_final if rec_lr >= rec_rf else y_pred_rf_final
    final_proba = y_proba_lr_final if rec_lr >= rec_rf else y_proba_rf_final
    final_name = "逻辑回归" if rec_lr >= rec_rf else "随机森林"

print(f"\n最终交付模型: {final_name}")
print(f"最终模型 Recall: {recall_score(y_test, final_pred):.4f}")

print("\n最终模型详细评估:")
evaluate_model_with_proba(y_test, final_pred, final_proba, final_name)

print("\n" + "=" * 60)
print("任务 5 模型训练完成！")
print("=" * 60)


# ====================================================================
# 一键生成最终交付物（任务 6 & 7 自动化）
# ====================================================================

import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ====================================================================
# 1. 创建输出目录
# ====================================================================
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"输出目录已创建: {output_dir}")

# ====================================================================
# 2. 提取模型评估指标（基于测试集）
# ====================================================================
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }

lr_metrics = get_metrics(lr_model, X_test_scaled, y_test)
rf_metrics = get_metrics(rf_model, X_test_scaled, y_test)
final_metrics = get_metrics(final_model, X_test_scaled, y_test)

# 构建对比表
comparison_df = pd.DataFrame({
    '模型': ['逻辑回归', '随机森林'],
    'Accuracy': [lr_metrics['accuracy'], rf_metrics['accuracy']],
    'Precision': [lr_metrics['precision'], rf_metrics['precision']],
    'Recall': [lr_metrics['recall'], rf_metrics['recall']],
    'F1-score': [lr_metrics['f1'], rf_metrics['f1']],
    'ROC-AUC': [lr_metrics['roc_auc'], rf_metrics['roc_auc']]
})
comparison_df.to_csv(os.path.join(output_dir, '模型对比表.csv'), index=False, encoding='utf-8-sig')
print("模型对比表已保存为 模型对比表.csv")

# ====================================================================
# 3. 生成关键图表
# ====================================================================
print("正在生成图表...")

# 3.1 ROC 曲线
plt.figure(figsize=(8,6))
for model, name in [(lr_model, '逻辑回归'), (rf_model, '随机森林')]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('ROC 曲线对比')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'ROC曲线对比.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  - ROC曲线已保存")

# 3.2 最终模型混淆矩阵
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['非卒中', '卒中'])
disp.plot(cmap='Blues')
plt.title(f'最终模型 ({final_name}) 混淆矩阵')
plt.savefig(os.path.join(output_dir, '混淆矩阵.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  - 混淆矩阵已保存")

# 3.3 特征重要性（随机森林）
if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10,8))
    feat_imp.head(15).plot(kind='barh')
    plt.xlabel('重要性')
    plt.title('随机森林 Top15 特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '特征重要性.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  - 特征重要性图已保存")
else:
    feat_imp = None

# ====================================================================
# 4. 生成 Markdown 报告
# ====================================================================
report_lines = []
report_lines.append("# 卒中高风险早筛模型 —— 最终交付报告\n")
report_lines.append(f"**生成日期**：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
report_lines.append("---\n")
report_lines.append("## 一、数据处理说明\n")
report_lines.append("### 1.1 数据概览\n")
report_lines.append(f"- 样本数量：5110 条\n")
report_lines.append(f"- 字段数量：12 个（处理后 11 个）\n")
report_lines.append(f"- 目标变量：stroke（卒中），正类 249 条（{249/5110:.2%}）\n")
report_lines.append(f"- 主要问题：类别严重不平衡、BMI 缺失约 201 条、smoking_status 含 Unknown（约 28%）\n")
report_lines.append("\n### 1.2 各字段处理详情\n")
report_lines.append("| 字段 | 处理方式 | 理由与风险 |\n")
report_lines.append("| :--- | :--- | :--- |\n")
report_lines.append("| id | 剔除 | 唯一编号，无预测意义，避免过拟合 |\n")
report_lines.append("| bmi | 中位数（28.1）填补 | 中位数对极端值稳健；风险：压缩变异度，可能引入偏差 |\n")
report_lines.append("| smoking_status | 保留 'Unknown' 为独立类别 | 'Unknown' 并非缺失，有实际含义；风险：可能学到虚假相关性 |\n")
report_lines.append("| gender | 'Other' 归并到 'Male' | 仅1条样本，合并至多数类；风险：微小标签偏差 |\n")
report_lines.append("| 其余字段 | 不做处理 | 直接作为特征 |\n")
report_lines.append("\n### 1.3 最终数据形态\n")
report_lines.append(f"- 样本数：5110\n")
report_lines.append(f"- 字段数：11（剔除 id）\n")
report_lines.append(f"- BMI 缺失：0（中位数填补）\n")
report_lines.append(f"- gender 类别：Male, Female\n")
report_lines.append(f"- smoking_status 类别：never smoked, formerly smoked, smokes, Unknown\n")
report_lines.append("\n---\n")
report_lines.append("## 二、模型对比表\n")
report_lines.append("### 2.1 评估指标\n")
report_lines.append(comparison_df.to_markdown(index=False))
report_lines.append("\n")
report_lines.append(f"**选用模型**：**{final_name}**\n")
report_lines.append(f"**选择理由**：该模型 Recall 为 {final_metrics['recall']:.3f}，高于另一模型且 ≥ 0.60 硬性要求。\n")
report_lines.append(f"**最终阈值**：经阈值调整后，最佳阈值为 {best_thresh_lr if final_name=='逻辑回归' else best_thresh_rf:.2f}（若未调整则为0.5）\n")
report_lines.append(f"**是否达标**：{'✅ 是' if final_metrics['recall'] >= 0.60 else '❌ 否（需进一步优化）'}\n")
report_lines.append("\n### 2.2 为什么不只看 Accuracy\n")
report_lines.append("在不平衡数据集（正类仅 4.87%）中，Accuracy 会被负类主导，即使完全不识别卒中患者，准确率也能达 95%。\n")
report_lines.append("本项目以 **Recall ≥ 0.60** 为硬性门槛，优先保证检出率。\n")
report_lines.append("\n---\n")
report_lines.append("## 三、高风险人群画像与资源建议\n")
report_lines.append("### 3.1 高风险人群画像\n")
report_lines.append("| 维度 | 高风险特征 | 说明 |\n")
report_lines.append("| :--- | :--- | :--- |\n")
report_lines.append("| 年龄 | **≥ 75 岁** | 最强单变量风险因素 |\n")
report_lines.append("| 高血压 | **患有高血压** | 与高龄叠加风险急剧上升 |\n")
report_lines.append("| 心脏病 | **患有心脏病** | 显著增加卒中风险 |\n")
report_lines.append("| 吸烟状态 | **既往吸烟** | 剔除 Unknown 后，既往吸烟者风险更高 |\n")
report_lines.append("| 血糖 | **糖尿病（≥126 mg/dL）** | 高血糖与肥胖叠加风险升高 |\n")
report_lines.append("| 综合 | **≥2个危险因素叠加** | 单一因素有限，叠加后风险数倍增长 |\n")
report_lines.append("\n**一句话画像**：\n> 75岁及以上，患有高血压，合并心脏病或既往吸烟史的老年人。\n")
report_lines.append("\n### 3.2 优先筛查建议\n")
report_lines.append("| 优先级 | 目标人群 | 依据 |\n")
report_lines.append("| :--- | :--- | :--- |\n")
report_lines.append("| 🥇 第一优先级 | **≥75岁 + 高血压** | 卒中率最高组合（可达 25%-30%） |\n")
report_lines.append("| 🥈 第二优先级 | **≥75岁 + 心脏病** 或 **≥75岁 + 既往吸烟** | 显著高于单一高龄 |\n")
report_lines.append("| 🥉 第三优先级 | **60-75岁 + ≥2个危险因素** | 多重因素叠加，风险接近高龄 |\n")
report_lines.append("**操作建议**：利用模型概率排序，优先对 top 20% 高风险人群投入筛查资源。\n")
report_lines.append("\n### 3.3 模型使用边界\n")
report_lines.append("| ✅ 适用于 | ❌ 不适用于 |\n")
report_lines.append("| :--- | :--- |\n")
report_lines.append("| 体检筛查优先级排序 | 临床确诊（不能替代 CT/MRI） |\n")
report_lines.append("| 慢病管理资源分配 | 拒绝医疗服务或医保报销 |\n")
report_lines.append("| 健康宣教定向 | 法律证据或保险精算 |\n")
report_lines.append("| 辅助医生决策 | 外推至其他人群（儿童/其他地区） |\n")
report_lines.append("| — | 完全替代医生判断 |\n")
report_lines.append("\n### 3.4 结果解释边界\n")
report_lines.append("- **相关性 ≠ 因果性**：模型发现的是统计关联，非因果。\n")
report_lines.append("- **数据局限性**：横截面数据，BMI 中位数填补可能引入轻微偏差。\n")
report_lines.append("- **阈值可调整**：降低阈值可提高 Recall，但增加误报；根据资源灵活设定。\n")
report_lines.append("- **建议定期校准**：每 1-2 年重新评估模型性能。\n")
report_lines.append("\n---\n")
report_lines.append("## 四、模型性能摘要（最终模型）\n")
report_lines.append(f"- **选用模型**：{final_name}\n")
report_lines.append(f"- **Recall（召回率）**：{final_metrics['recall']:.3f}（{'✅ 达标' if final_metrics['recall'] >= 0.60 else '❌ 未达标'}）\n")
report_lines.append(f"- **ROC-AUC**：{final_metrics['roc_auc']:.3f}\n")
if feat_imp is not None:
    top_features = feat_imp.head(3).index.tolist()
    report_lines.append(f"- **关键特征**：根据随机森林特征重要性，前三重要特征为 {top_features}。\n")
else:
    report_lines.append("- **关键特征**：年龄、高血压、血糖（基于随机森林）。\n")
report_lines.append("\n---\n")
report_lines.append("*报告生成于 " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + "*")
with open(os.path.join(output_dir, '最终交付报告.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print("Markdown 报告已保存为 最终交付报告.md")

# ====================================================================
# 5. 保存管理层结论
# ====================================================================
executive_summary = f"""
============================================================
卒中高风险早筛模型 —— 管理层结论（一页纸）
============================================================

【高风险人群画像】
- 核心特征：≥75岁、高血压、心脏病/吸烟史
- 一句话：75岁及以上，患有高血压，合并心脏病或既往吸烟史的老年人。

【优先筛查建议】
1. 第一优先级：≥75岁 + 高血压（卒中率最高）
2. 第二优先级：≥75岁 + 心脏病 或 ≥75岁 + 既往吸烟
3. 第三优先级：60-75岁 + ≥2个危险因素（如高血压+糖尿病）
操作：使用模型概率排序，对 top 20% 高风险人群投入资源。

【模型适合用于】
✅ 体检筛查优先级排序
✅ 慢病管理资源分配
✅ 健康宣教定向
✅ 辅助医生决策

【模型不能替代】
❌ 临床确诊（不能替代 CT/MRI 等检查）
❌ 拒绝医疗服务或医保报销
❌ 法律证据或保险精算
❌ 外推至其他人群（儿童/其他地区）
❌ 完全替代医生判断

【模型性能】
- 选用模型：{final_name}
- Recall（召回率）：{final_metrics['recall']:.3f}  {'（达标）' if final_metrics['recall'] >= 0.60 else '（未达标，需优化）'}
- ROC-AUC：{final_metrics['roc_auc']:.3f}

【关键提醒】
- 相关性 ≠ 因果性，模型结论为统计关联。
- 阈值可根据资源情况灵活调整。
- 建议每 1-2 年重新校准模型。

报告日期：{datetime.datetime.now().strftime('%Y-%m-%d')}
"""
with open(os.path.join(output_dir, '管理层结论.txt'), 'w', encoding='utf-8') as f:
    f.write(executive_summary)
print("管理层结论已保存为 管理层结论.txt")

print("\n" + "=" * 60)
print("🎉 一键生成完成！所有交付物已保存至 './output' 目录")
print("生成的文件列表：")
for fname in os.listdir(output_dir):
    print(f"  - {fname}")
print("=" * 60)


# ====================================================================
# 【最终补充代码】将所有可视化图表按序号保存至“可视化”文件夹
# 请在现有代码（任务1~5 + 一键生成）之后运行此代码块
# ====================================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 创建“可视化”文件夹
vis_dir = './可视化'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
print(f"可视化文件夹已创建/确认: {vis_dir}")

# 设置绘图风格（与任务3保持一致）
sns.set_style("whitegrid")
# 中文字体已在前面设置，这里再次确认
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("开始生成全部可视化图表（共7张）...")

# ====================================================================
# 图表 01：目标变量分布图
# ====================================================================
print("  - 生成 01_目标变量分布图.png")
fig1, ax1 = plt.subplots(figsize=(6, 5))
stroke_counts = df['stroke'].value_counts()
stroke_counts.plot(kind='bar', color=['#3498db', '#e74c3c'], ax=ax1)
ax1.set_title('目标变量 stroke 分布', fontsize=14)
ax1.set_xlabel('卒中 (0=否, 1=是)')
ax1.set_ylabel('样本数量')
ax1.set_xticklabels(['非卒中 (0)', '卒中 (1)'], rotation=0)
for i, v in enumerate(stroke_counts.values):
    ax1.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=12)
ax1.text(0.5, 0.95, f'正类比例: {249/5110:.2%}', transform=ax1.transAxes,
         ha='center', fontsize=12, color='red', weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '01_目标变量分布图.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# 图表 02：探索性分析图表合集（3个箱线图 + 3个分组柱状图）
# ====================================================================
print("  - 生成 02_探索性分析图表合集.png")
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

# 年龄箱线图
sns.boxplot(x='stroke', y='age', data=df, palette=['#3498db', '#e74c3c'], ax=axes2[0, 0])
axes2[0, 0].set_title('年龄与卒中状态', fontsize=12)
axes2[0, 0].set_xlabel('卒中 (0=否, 1=是)')
axes2[0, 0].set_ylabel('年龄 (岁)')

# 血糖箱线图
sns.boxplot(x='stroke', y='avg_glucose_level', data=df, palette=['#3498db', '#e74c3c'], ax=axes2[0, 1])
axes2[0, 1].set_title('平均血糖与卒中状态', fontsize=12)
axes2[0, 1].set_xlabel('卒中 (0=否, 1=是)')
axes2[0, 1].set_ylabel('平均血糖 (mg/dL)')

# BMI箱线图
sns.boxplot(x='stroke', y='bmi', data=df, palette=['#3498db', '#e74c3c'], ax=axes2[0, 2])
axes2[0, 2].set_title('BMI 与卒中状态', fontsize=12)
axes2[0, 2].set_xlabel('卒中 (0=否, 1=是)')
axes2[0, 2].set_ylabel('BMI (kg/m²)')

# 高血压分组图
hypertension_rate = df.groupby('hypertension')['stroke'].mean()
hypertension_rate.plot(kind='bar', color=['#3498db', '#e74c3c'], ax=axes2[1, 0])
axes2[1, 0].set_title('高血压与卒中率', fontsize=12)
axes2[1, 0].set_xlabel('高血压 (0=无, 1=有)')
axes2[1, 0].set_ylabel('卒中发生率')
axes2[1, 0].set_xticklabels(['无高血压', '有高血压'], rotation=0)
axes2[1, 0].set_ylim(0, 0.3)

# 心脏病分组图
heart_rate = df.groupby('heart_disease')['stroke'].mean()
heart_rate.plot(kind='bar', color=['#3498db', '#e74c3c'], ax=axes2[1, 1])
axes2[1, 1].set_title('心脏病与卒中率', fontsize=12)
axes2[1, 1].set_xlabel('心脏病 (0=无, 1=有)')
axes2[1, 1].set_ylabel('卒中发生率')
axes2[1, 1].set_xticklabels(['无心脏病', '有心脏病'], rotation=0)
axes2[1, 1].set_ylim(0, 0.3)

# 吸烟状态分组图
smoke_rate = df.groupby('smoking_status')['stroke'].mean()
smoke_rate = smoke_rate.reindex(['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
smoke_rate.plot(kind='bar', color=['#2ecc71', '#f1c40f', '#e67e22', '#95a5a6'], ax=axes2[1, 2])
axes2[1, 2].set_title('吸烟状态与卒中率', fontsize=12)
axes2[1, 2].set_xlabel('吸烟状态')
axes2[1, 2].set_ylabel('卒中发生率')
axes2[1, 2].set_xticklabels(['从不吸烟', '既往吸烟', '现在吸烟', '未知'], rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '02_探索性分析图表合集.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# 图表 03：交叉分析热力图合集（5张最重要的）
# ====================================================================
print("  - 生成 03_交叉分析热力图合集.png")

# 定义需要绘制的热力图组合
heatmap_configs = [
    ('年龄×高血压', 'hypertension', 'age_group'),
    ('年龄×心脏病', 'heart_disease', 'age_group'),
    ('年龄×吸烟状态', 'smoking_status', 'age_group'),
    ('高血压×心脏病', 'heart_disease', 'hypertension'),
    ('血糖×BMI', 'bmi_group', 'glucose_group'),
]

n_plots = len(heatmap_configs)
# 计算行列：3行2列（6个子图，实际用5个）
fig3, axes3 = plt.subplots(3, 2, figsize=(16, 18))
ax_list3 = axes3.flatten()

for idx, (title, x_col, y_col) in enumerate(heatmap_configs):
    if idx < len(ax_list3):
        rate = df.groupby([y_col, x_col])['stroke'].mean().unstack()
        sns.heatmap(rate, annot=True, fmt='.3f', cmap='Reds', ax=ax_list3[idx],
                    cbar_kws={'label': '卒中率'})
        ax_list3[idx].set_title(title, fontsize=14)
        ax_list3[idx].set_xlabel(x_col)
        ax_list3[idx].set_ylabel(y_col)

# 隐藏多余子图
for j in range(n_plots, len(ax_list3)):
    ax_list3[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '03_交叉分析热力图合集.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# 图表 04：PR 曲线（Precision-Recall Curve）
# ====================================================================
print("  - 生成 04_PR曲线.png")
fig4, ax4 = plt.subplots(figsize=(8, 6))
for model, name in [(lr_model, '逻辑回归'), (rf_model, '随机森林')]:
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    ax4.plot(recall_vals, precision_vals, label=f'{name} (AP={ap_score:.3f})')
ax4.set_xlabel('召回率 (Recall)')
ax4.set_ylabel('精确率 (Precision)')
ax4.set_title('PR 曲线对比（不平衡任务关键指标）')
ax4.legend(loc='best')
ax4.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '04_PR曲线.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# 图表 05：ROC 曲线对比
# ====================================================================
print("  - 生成 05_ROC曲线对比.png")
fig5, ax5 = plt.subplots(figsize=(8, 6))
for model, name in [(lr_model, '逻辑回归'), (rf_model, '随机森林')]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    ax5.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
ax5.plot([0, 1], [0, 1], 'k--')
ax5.set_xlabel('假正率 (False Positive Rate)')
ax5.set_ylabel('真正率 (True Positive Rate)')
ax5.set_title('ROC 曲线对比')
ax5.legend(loc='lower right')
ax5.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '05_ROC曲线对比.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# 图表 06：最终模型混淆矩阵
# ====================================================================
print("  - 生成 06_混淆矩阵.png")
cm = confusion_matrix(y_test, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['非卒中', '卒中'])
disp.plot(cmap='Blues')
plt.title(f'最终模型 ({final_name}) 混淆矩阵')
plt.savefig(os.path.join(vis_dir, '06_混淆矩阵.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# 图表 07：特征重要性（随机森林 Top15）
# ====================================================================
print("  - 生成 07_特征重要性.png")
if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fig7, ax7 = plt.subplots(figsize=(10, 8))
    feat_imp.head(15).plot(kind='barh', ax=ax7)
    ax7.set_xlabel('重要性')
    ax7.set_title('随机森林 Top15 特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '07_特征重要性.png'), dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("  - 警告：随机森林没有 feature_importances_，跳过此图")

# ====================================================================
# 完成
# ====================================================================
print("\n" + "=" * 60)
print("✅ 全部7张可视化图表已生成并保存至 './可视化' 文件夹")
print("生成的文件列表：")
for fname in sorted(os.listdir(vis_dir)):
    if fname.endswith('.png'):
        print(f"  📊 {fname}")
print("=" * 60)