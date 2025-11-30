# 加载必要的包
library(MASS)         # 用于生成多元正态分布
library(glmnet)       # Lasso和弹性网
library(ncvreg)       # SCAD
library(pROC)         # ROC曲线
library(Metrics)      # 计算AUC和其他指标
library(ggplot2)      # 用于绘图
library(foreach)      # 用于并行计算
library(doParallel)   # 用于并行计算

# 设置随机种子
set.seed(123)

# 参数设置
n <- 300              # 样本量
p_values <- c(50, 100, 150, 200, 400, 600)  # 维数
rho_values <- c(0.3, 0.6)  # 相关系数
num_simulations <- 200  # 模拟次数

# 真实回归系数
beta_true <- c(-1.5, 1, 1.2, -0.8, 0.6, rep(0, max(p_values) - 5))

# 用于存储结果
results <- list()

# 创建多核集群
cl <- makeCluster(detectCores() - 5)  # 使用所有可用核心，减去1
doParallel::registerDoParallel(cl)

# 并行执行模拟
results_list <- foreach(rho = rho_values, .combine = 'rbind',.packages = c('MASS','glmnet','ncvreg','pROC','Metrics')) %:%
  foreach(p = p_values, .combine = 'rbind',.packages = c('MASS','glmnet','ncvreg','pROC','Metrics')) %dopar% {

sim_results <- run_simulation(rho, p)

results<-data.frame(
      rho = rho,
      p = p,
      C_Lasso = mean(sim_results$C_Lasso),
      IC_Lasso = mean(sim_results$IC_Lasso),
      EE_Lasso = mean(sim_results$EE_Lasso),
      errRate_Lasso = mean(sim_results$errRate_Lasso),
      AUC_Lasso = mean(sim_results$AUC_Lasso),
      
      C_Lasso_admm = mean(sim_results$C_Lasso_admm),
      IC_Lasso_admm = mean(sim_results$IC_Lasso_admm),
      EE_Lasso_admm = mean(sim_results$EE_Lasso_admm),
      errRate_Lasso_admm = mean(sim_results$errRate_Lasso_admm),
      AUC_Lasso_admm = mean(sim_results$AUC_Lasso_admm),
      
      C_Ridge = mean(sim_results$C_Ridge),
      IC_Ridge = mean(sim_results$IC_Ridge),
      EE_Ridge = mean(sim_results$EE_Ridge),
      errRate_Ridge = mean(sim_results$errRate_Ridge),
      AUC_Ridge = mean(sim_results$AUC_Ridge),
      
      C_SCAD = mean(sim_results$C_SCAD),
      IC_SCAD = mean(sim_results$IC_SCAD),
      EE_SCAD = mean(sim_results$EE_SCAD),
      errRate_SCAD = mean(sim_results$errRate_SCAD),
      AUC_SCAD = mean(sim_results$AUC_SCAD),
      
      C_SCAD_lqa = mean(sim_results$C_SCAD_lqa),
      IC_SCAD_lqa = mean(sim_results$IC_SCAD_lqa),
      EE_SCAD_lqa = mean(sim_results$EE_SCAD_lqa),
      errRate_SCAD_lqa = mean(sim_results$errRate_SCAD_lqa),
      AUC_SCAD_lqa = mean(sim_results$AUC_SCAD_lqa),
      
      C_SCAD_mm = mean(sim_results$C_SCAD_mm),
      IC_SCAD_mm = mean(sim_results$IC_SCAD_mm),
      EE_SCAD_mm = mean(sim_results$EE_SCAD_mm),
      errRate_SCAD_mm = mean(sim_results$errRate_SCAD_mm),
      AUC_SCAD_mm = mean(sim_results$AUC_SCAD_mm),
      
      C_SCAD_lla = mean(sim_results$C_SCAD_lla),
      IC_SCAD_lla = mean(sim_results$IC_SCAD_lla),
      EE_SCAD_lla = mean(sim_results$EE_SCAD_lla),
      errRate_SCAD_lla = mean(sim_results$errRate_SCAD_lla),
      AUC_SCAD_lla = mean(sim_results$AUC_SCAD_lla),
      

      C_Adaptive_Lasso = mean(sim_results$C_Adaptive_Lasso),
      IC_Adaptive_Lasso = mean(sim_results$IC_Adaptive_Lasso),
      EE_Adaptive_Lasso = mean(sim_results$EE_Adaptive_Lasso),
      errRate_Adaptive_Lasso = mean(sim_results$errRate_Adaptive_Lasso),
      AUC_Adaptive_Lasso = mean(sim_results$AUC_Adaptive_Lasso),
      
      C_Elastic_Net = mean(sim_results$C_Elastic_Net),
      IC_Elastic_Net = mean(sim_results$IC_Elastic_Net),
      EE_Elastic_Net = mean(sim_results$EE_Elastic_Net),
      errRate_Elastic_Net = mean(sim_results$errRate_Elastic_Net),
      AUC_Elastic_Net = mean(sim_results$AUC_Elastic_Net)
    )
 }
# 停止集群
stopCluster(cl)

# 输出结果
print(results_list)

