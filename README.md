# 高维逻辑回归的惩罚估计算法

本项目用 R 探索高维二分类逻辑回归中的惩罚估计与变量选择，包含自定义优化算法、调用现有 R 包的方法比较，以及配套的推导与模拟报告。

## 目录与入口

| 路径 | 内容 / 入口 |
| --- | --- |
| [`code/LASSO-ADMM.R`](code/LASSO-ADMM.R) | LASSO 的 ADMM 实现：`admm_logistic_lasso(X, y, lambda)`。 |
| [`code/SCAD-LQA.R`](code/SCAD-LQA.R) | SCAD 的局部二次近似：`scad_lqa(X, y, lambda)`。 |
| [`code/SCAD-MM.R`](code/SCAD-MM.R) | SCAD 的 MM 实现：`scad_mm(X, y, lambda)`。 |
| [`code/SCAD-LLA.R`](code/SCAD-LLA.R) | SCAD 的局部线性近似：`scad_lla(X, y, lambda)`。 |
| [`code/数值模拟函数.R`](code/数值模拟函数.R) | 定义 `run_simulation(rho, p)`；生成数据、拟合模型并返回各次模拟的指标。依赖预先加载的算法函数及全局变量 `n`、`num_simulations`、`beta_true`。 |
| [`code/数值模拟代码.R`](code/数值模拟代码.R) | 设置参数、并行调用模拟函数，汇总并打印 `results_list`。 |
| [`code/ROC模拟.R`](code/ROC模拟.R) | 定义 `roc_simulation(rho, p)`，合并重复模拟的测试预测，计算并绘制 ROC 曲线；包含并行执行代码。 |
| [`Picture/`](Picture/) | 12 个按 `rho` 和 `p` 命名的已保存 ROC 图 PDF。 |
| [`高维逻辑回归的惩罚估计算法.pdf`](高维逻辑回归的惩罚估计算法.pdf) | 方法推导、模拟设置、数值结果表、ROC 图与参考文献。 |

模拟代码调用 `glmnet` 比较 LASSO、Ridge、自适应 LASSO，调用 `ncvreg` 拟合 SCAD，并调用上述四个自定义算法。另有标为 **Elastic Net** 的分支，但该分支未显式设置 `alpha`，使用前需核对包的默认值与预期混合比例。

## 依赖

需要 R。下面的串行示例使用 `MASS`、`glmnet`、`ncvreg`、`pROC`；两个完整模拟入口还加载 `Metrics`、`ggplot2`、`foreach`、`doParallel`。仓库未提供 R 或包版本锁定文件。

```r
# 在 R 控制台安装示例所需的包（只需安装一次）
install.packages(c("MASS", "glmnet", "ncvreg", "pROC"))
# 如需使用完整模拟入口，再安装：
# install.packages(c("Metrics", "ggplot2", "foreach", "doParallel"))
```

## 最短运行路径：单个设置的串行模拟

下载或克隆仓库后，将 R 的工作目录设为仓库根目录，在 R 控制台运行以下代码。此示例直接调用已有函数，避免启动完整并行任务；只做一次模拟，用于尝试调用流程，不用于评价方法优劣。

```r
library(MASS)
library(glmnet)
library(ncvreg)
library(pROC)

for (file in c("LASSO-ADMM.R", "SCAD-LQA.R", "SCAD-MM.R", "SCAD-LLA.R")) {
  source(file.path("code", file), encoding = "UTF-8")
}
source("code/数值模拟函数.R", encoding = "UTF-8")

set.seed(123)
n <- 300
p <- 50
num_simulations <- 1
beta_true <- c(-1.5, 1, 1.2, -0.8, 0.6, rep(0, p - 5))

result <- run_simulation(rho = 0.3, p = p)
print(sapply(result, mean))
```

`C` 为选中的真实非零系数个数，`IC` 为选中的零系数个数，`EE` 为系数估计的 L2 误差，`errRate` 为测试集分类错误率，`AUC` 为测试集 ROC 曲线下面积。代码使用 70% / 30% 的随机训练 / 测试划分；变量选择与系数误差指标使用全样本拟合，预测指标使用训练集拟合。

## 完整实验与 PDF 的关系

完整入口设置 `n = 300`、`p = 50, 100, 150, 200, 400, 600`、`rho = 0.3, 0.6`；协变量为协方差 `Sigma[i, j] = rho^abs(i - j)` 的多元正态数据，响应按逻辑回归概率生成。PDF 第 1 节说明方法，第 2 节展示对应主题的模拟设置、结果表与 ROC 图；PDF 中的 LASSO Newton-Raphson 讨论没有同名独立脚本。

使用完整入口前需处理以下现有约束：

- 两个入口均未自动 `source()` 自定义算法；数值模拟还需加载 `数值模拟函数.R`，并确保并行 worker 可访问所需函数与参数。
- 数值模拟使用 `makeCluster(detectCores() - 5)`，需根据机器调整为有效的 worker 数；ROC 入口的 `.packages = 'your_package_name'` 是待替换的占位符，末尾 `windows()` 绘图调用也需按平台调整。
- PDF 第 2.1 节写明重复 **500** 次，当前数值模拟入口设为 **200** 次，ROC 入口设为 **100** 次；两者均有 `set.seed(123)`，但未显式配置并行随机数流。
- 数值模拟入口打印结果，ROC 入口绘图；当前脚本未提供自动写出报告表格或 `Picture/` 中全部 PDF 的流程。

本 README 依据现有源码与 PDF 核对；整理时未执行 R 仿真。实际运行、软件版本兼容性及现有报告图表的精确复现尚未验证。
