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
num_simulations <- 100  # 模拟次数

# 真实回归系数
beta_true <- c(-1.5, 1, 1.2, -0.8, 0.6, rep(0, max(p_values) - 5))
# 定义sigmoid函数
sig<-function(x){
  return(exp(x) / (1 + exp(x)))
}

roc_simulation <- function(rho, p) {
  # 创建一个空的列表存储 ROC 数据
  roc_data <- list()
  y_test_total<-c()
  lasso_pred<-c()
  lasso_admm_pred<-c()
  ridge_pred<-c()
  scad_pred<-c()
  scad_lqa_pred<-c()
  scad_mm_pred<-c()
  scad_lla_pred<-c()
  adaptive_lasso_pred<-c()
  elastic_net_pred<-c()
  for (sim in 1:num_simulations) {
    # 生成协变量
    Sigma <- outer(1:p, 1:p, function(i, j) rho^(abs(i - j)))
    x <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
    
    # 生成响应变量
    pi_x <- exp(beta_true[1:p] %*% t(x)) / (1 + exp(beta_true[1:p] %*% t(x)))  
    y <- rbinom(n, 1, pi_x)
    
    # 划分训练集和测试集
    train_indices <- sample(1:n, size = 0.7 * n)
    x_train <- x[train_indices, ]
    y_train <- y[train_indices]
    x_test <- x[-train_indices, ]
    y_test <- y[-train_indices]
    y_test_total <- c(y_test_total,y_test)
    # Lasso
    cv_lasso_model_train <- cv.glmnet(x_train, y_train, alpha = 1,family="binomial")
    lasso_model_train <- glmnet(x_train, y_train, alpha = 1,family="binomial")
    lasso_pred <- c(lasso_pred,predict(lasso_model_train , s = cv_lasso_model_train$lambda.min, newx = x_test, type = "response"))
    
    # Lasso-admm
    lasso_admm_model_train_coef<-admm_logistic_lasso(x_train, y_train, cv_lasso_model_train$lambda.min)
    lasso_admm_pred <- c(lasso_admm_pred,sig(x_test%*%lasso_admm_model_train_coef))
    
    # Ridge
    cv_ridge_model_train <- cv.glmnet(x_train, y_train, alpha = 0,family="binomial")
    ridge_model_train <- glmnet(x_train, y_train, alpha = 0,family="binomial")
    ridge_pred <- c(ridge_pred,predict(ridge_model_train, s = cv_ridge_model_train$lambda.min, newx = x_test, type = "response"))
    
    # SCAD
    #scad的预测 
    cv_scad_model_train <- cv.ncvreg(x_train, y_train, penalty = "SCAD", family="binomial")
    scad_model_train <- ncvreg(x_train, y_train, penalty = "SCAD", family="binomial")
    # 找到最接近的 lambda 值的位置
    available_lambdas <- scad_model_train$lambda
    closest_lambda_index <- which.min(abs(available_lambdas - cv_scad_model_train$lambda.min))
    cv_scad_model_train_lambdamin_index<-closest_lambda_index
    #找到的最接近的 lambda 值的训练模型上的系数向量
    coef_scad_model_train<-coef(scad_model_train)[,cv_scad_model_train_lambdamin_index][-1]
    scad_pred <- c(scad_pred,sig(coef_scad_model_train%*% t(x_test)))
    
    scad_lambda=0.1
    # scad_lambda=cv_scad_model$lambda.min
    #scad-lqa
    scad_lqa_model_train_coef <- scad_lqa(x_train,y_train,scad_lambda)
    scad_lqa_pred <- c(scad_lqa_pred,sig(x_test %*% scad_lqa_model_train_coef))
    
    #scad-mm
    scad_mm_model_train_coef <- scad_mm(x_train,y_train,scad_lambda)
    scad_mm_pred <- c(scad_mm_pred,sig(x_test %*% scad_mm_model_train_coef))
    
    # scad-lla
    scad_lla_model_train_coef <- scad_lla(x_train, y_train, scad_lambda)
    scad_lla_pred <- c(scad_lla_pred,sig(x_test %*% scad_lla_model_train_coef))
    
    # 自适应Lasso
    coef_lasso <- coef(lasso_model_train, s = cv_lasso_model_train$lambda.min)[-1]
    weights <- 1 / abs(coef_lasso)
    weights[is.infinite(weights)] <- 10^5   # 处理零系数
    cv_adaptive_lasso_model_train <- cv.glmnet(x_train, y_train, alpha = 1, penalty.factor = weights,family="binomial")
    adaptive_lasso_model_train <- glmnet(x_train, y_train, alpha = 1, penalty.factor = weights,family="binomial")
    adaptive_lasso_pred <- c(adaptive_lasso_pred,predict(adaptive_lasso_model_train, s = cv_adaptive_lasso_model_train$lambda.min, newx = x_test, type = "response"))
    
    # 弹性网
    cv_elastic_net_model_train <- cv.glmnet(x_train, y_train, family="binomial")
    elastic_net_model_train <- glmnet(x_train, y_train, family="binomial")
    elastic_net_pred <- c(elastic_net_pred,predict(elastic_net_model_train, s = cv_elastic_net_model_train$lambda.min, newx = x_test, type = "response"))
  }
    
    # 计算指标
    y_test=y_test_total
    # Lasso
    lasso_pred_vector <- as.vector(lasso_pred)
    roc_data[["LASSO"]] <- pROC::roc(y_test, lasso_pred_vector)  # 计算ROC曲线

    # Lasso-admm
    lasso_admm_pred_vector <- as.vector(lasso_admm_pred)
    roc_data[["LASSO_ADMM"]] <- pROC::roc(y_test, lasso_admm_pred_vector)  # 计算ROC曲线

    # Ridge
   ridge_pred_vector <- as.vector(ridge_pred)
   roc_data[["Ridge"]] <- pROC::roc(y_test, ridge_pred_vector)  # 计算ROC曲线
 
    # SCAD
    scad_pred_vector <- as.vector(scad_pred)
    roc_data[["SCAD"]] <- pROC::roc(y_test, scad_pred_vector)  # 计算ROC曲线
    
    # SCAD_lqa
    scad_lqa_pred_vector <- as.vector(scad_lqa_pred)
    roc_data[["SCAD_LQA"]] <- pROC::roc(y_test, scad_lqa_pred_vector)  # 计算ROC曲线

    # SCAD_mm
    scad_mm_pred_vector <- as.vector(scad_mm_pred)
    roc_data[["SCAD_MM"]] <- pROC::roc(y_test, scad_mm_pred_vector)  # 计算ROC曲线

    # SCAD_lla
    scad_lla_pred_vector <- as.vector(scad_lla_pred)
    roc_data[["SCAD_LLA"]] <- pROC::roc(y_test, scad_lla_pred_vector)  # 计算ROC曲线
   
    # 自适应Lasso
    adaptive_lasso_pred_vector <- as.vector(adaptive_lasso_pred)
    roc_data[["Adaptive_LASSO"]] <- pROC::roc(y_test, adaptive_lasso_pred_vector)  # 计算ROC曲线
    
    # 弹性网
    elastic_net_pred_vector <- as.vector(elastic_net_pred)
    roc_data[["Elastic_Net"]] <- pROC::roc(y_test, elastic_net_pred_vector)  # 计算ROC曲线
  return(roc_data)
}
roc_data_list<-list()
# roc_data_list[["rho=0.3_p=50"]]<-roc_simulation(rho_values[1], p_values[1])
# roc_data_list[["rho=0.3_p=100"]]<-roc_simulation(rho_values[1], p_values[2])
# roc_data_list[["rho=0.3_p=150"]]<-roc_simulation(rho_values[1], p_values[3])
# roc_data_list[["rho=0.3_p=200"]]<-roc_simulation(rho_values[1], p_values[4])
# roc_data_list[["rho=0.3_p=400"]]<-roc_simulation(rho_values[1], p_values[5])
# roc_data_list[["rho=0.3_p=600"]]<-roc_simulation(rho_values[1], p_values[6])
# roc_data_list[["rho=0.6_p=50"]]<-roc_simulation(rho_values[2], p_values[1])
# roc_data_list[["rho=0.6_p=100"]]<-roc_simulation(rho_values[2], p_values[2])
# roc_data_list[["rho=0.6_p=150"]]<-roc_simulation(rho_values[2], p_values[3])
# roc_data_list[["rho=0.6_p=200"]]<-roc_simulation(rho_values[2], p_values[4])
# roc_data_list[["rho=0.6_p=400"]]<-roc_simulation(rho_values[2], p_values[5])
# roc_data_list[["rho=0.6_p=600"]]<-roc_simulation(rho_values[2], p_values[6])

# 加载必要的库
library(foreach)
library(doParallel)

# 注册并行后端
cl <- makeCluster(detectCores() - 1)  # 使用可用核心数 - 1
registerDoParallel(cl)

# 创建参数列表
params <- expand.grid(p = p_values,rho = rho_values)

# 使用 foreach 进行并行计算
roc_data_list <- foreach(i = 1:nrow(params), .packages = 'your_package_name') %dopar% {
  roc_simulation(params$rho[i], params$p[i])
}

# 停止并行后端
stopCluster(cl)

# 将结果命名
names(roc_data_list) <- paste0("rho=", params$rho, "_p=", params$p)


# 绘制 ROC 曲线
roc_data<-roc_data_list[["rho=0.3_p=50"]]
windows(width = 10, height = 10) 
plot(roc_data[[1]], col = "blue", main = "ROC Curves for Different Models(rho=0.3 p=50)")
for (i in 2:length(roc_data)) {
  plot(roc_data[[i]], add = TRUE, col = i)  # 添加到现有图形
}

# 添加对角线
abline(0, 1, lty = 2, col = "gray")

# 添加图例
legend("bottomright", legend = names(roc_data),
       col = 1:length(roc_data), lwd = 2)


