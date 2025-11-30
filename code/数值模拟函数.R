# 定义sigmoid函数
sig<-function(x){
  return(exp(x) / (1 + exp(x)))
}

# 定义模拟函数
run_simulation <- function(rho, p) {
  # 存储每种方法的各项指标
  results <- list(
    C_Lasso = numeric(num_simulations),
    IC_Lasso = numeric(num_simulations),
    EE_Lasso = numeric(num_simulations),
    errRate_Lasso = numeric(num_simulations),
    AUC_Lasso = numeric(num_simulations),
    
    C_Lasso_admm = numeric(num_simulations),
    IC_Lasso_admm = numeric(num_simulations),
    EE_Lasso_admm = numeric(num_simulations),
    errRate_Lasso_admm = numeric(num_simulations),
    AUC_Lasso_admm = numeric(num_simulations),
    
    C_Ridge = numeric(num_simulations),
    IC_Ridge = numeric(num_simulations),
    EE_Ridge = numeric(num_simulations),
    errRate_Ridge = numeric(num_simulations),
    AUC_Ridge = numeric(num_simulations),
    
    C_SCAD = numeric(num_simulations),
    IC_SCAD = numeric(num_simulations),
    EE_SCAD = numeric(num_simulations),
    errRate_SCAD = numeric(num_simulations),
    AUC_SCAD = numeric(num_simulations),
    
    C_SCAD_lqa = numeric(num_simulations),
    IC_SCAD_lqa = numeric(num_simulations),
    EE_SCAD_lqa = numeric(num_simulations),
    errRate_SCAD_lqa = numeric(num_simulations),
    AUC_SCAD_lqa = numeric(num_simulations),
    
    C_SCAD_mm = numeric(num_simulations),
    IC_SCAD_mm = numeric(num_simulations),
    EE_SCAD_mm = numeric(num_simulations),
    errRate_SCAD_mm = numeric(num_simulations),
    AUC_SCAD_mm = numeric(num_simulations),
    
    C_SCAD_lla = numeric(num_simulations),
    IC_SCAD_lla = numeric(num_simulations),
    EE_SCAD_lla = numeric(num_simulations),
    errRate_SCAD_lla = numeric(num_simulations),
    AUC_SCAD_lla = numeric(num_simulations),
    
    C_Adaptive_Lasso = numeric(num_simulations),
    IC_Adaptive_Lasso = numeric(num_simulations),
    EE_Adaptive_Lasso = numeric(num_simulations),
    errRate_Adaptive_Lasso = numeric(num_simulations),
    AUC_Adaptive_Lasso = numeric(num_simulations),
    
    C_Elastic_Net = numeric(num_simulations),
    IC_Elastic_Net = numeric(num_simulations),
    EE_Elastic_Net = numeric(num_simulations),
    errRate_Elastic_Net = numeric(num_simulations),
    AUC_Elastic_Net = numeric(num_simulations)
  )
  
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
    
    # Lasso
    cv_lasso_model <- cv.glmnet(x, y, alpha = 1,family="binomial") # 交叉验证确定最佳的 lambda 值
    lasso_model <- glmnet(x, y, alpha = 1,family="binomial")
    cv_lasso_model_train <- cv.glmnet(x_train, y_train, alpha = 1,family="binomial")
    lasso_model_train <- glmnet(x_train, y_train, alpha = 1,family="binomial")
    lasso_pred <- predict(lasso_model_train , s = cv_lasso_model_train$lambda.min, newx = x_test, type = "response")
    lasso_pred_class <- as.numeric(lasso_pred > 0.5)
    
    # Lasso-admm
    
    lasso_admm_model_coef<-admm_logistic_lasso(x, y, cv_lasso_model$lambda.min)
    lasso_admm_model_train_coef<-admm_logistic_lasso(x_train, y_train, cv_lasso_model_train$lambda.min)
    lasso_admm_pred <- sig(x_test%*%lasso_admm_model_train_coef)
    lasso_admm_pred_class <- as.numeric(lasso_admm_pred > 0.5)
    
    # Ridge
    cv_ridge_model <- cv.glmnet(x, y, alpha = 0,family="binomial")
    ridge_model <- glmnet(x, y, alpha = 0,family="binomial")
    cv_ridge_model_train <- cv.glmnet(x_train, y_train, alpha = 0,family="binomial")
    ridge_model_train <- glmnet(x_train, y_train, alpha = 0,family="binomial")
    ridge_pred <- predict(ridge_model_train, s = cv_ridge_model_train$lambda.min, newx = x_test, type = "response")
    ridge_pred_class <- as.numeric(ridge_pred > 0.5)
    
    # SCAD
    cv_scad_model <- cv.ncvreg(x, y, penalty = "SCAD", family="binomial")
    scad_model <- ncvreg(x, y, penalty = "SCAD", family="binomial")
    # 检查 ncvreg 模型的 lambda 值
    available_lambdas <- scad_model$lambda
    # 找到最接近的 lambda 值的位置
    closest_lambda_index <- which.min(abs(available_lambdas - cv_scad_model$lambda.min))
    cv_scad_model_lambdamin_index<-closest_lambda_index
    #scad的预测 
    cv_scad_model_train <- cv.ncvreg(x_train, y_train, penalty = "SCAD", family="binomial")
    scad_model_train <- ncvreg(x_train, y_train, penalty = "SCAD", family="binomial")
    # 找到最接近的 lambda 值的位置
    available_lambdas <- scad_model_train$lambda
    closest_lambda_index <- which.min(abs(available_lambdas - cv_scad_model_train$lambda.min))
    cv_scad_model_train_lambdamin_index<-closest_lambda_index
    #找到的最接近的 lambda 值的训练模型上的系数向量
    coef_scad_model_train<-coef(scad_model_train)[,cv_scad_model_train_lambdamin_index][-1]
    scad_pred <- sig(coef_scad_model_train%*% t(x_test))
    scad_pred_class <- as.numeric(scad_pred > 0.5)
    
    scad_lambda=0.1

    #scad-lqa
    scad_lqa_model_coef <- scad_lqa(x,y,scad_lambda)
    scad_lqa_model_train_coef <- scad_lqa(x_train,y_train,scad_lambda)
    scad_lqa_pred <- sig(x_test %*% scad_lqa_model_train_coef)
    scad_lqa_pred_class <- as.numeric(scad_lqa_pred > 0.5)
    
    #scad-mm
    scad_mm_model_coef <- scad_mm(x,y,scad_lambda)
    scad_mm_model_train_coef <- scad_mm(x_train,y_train,scad_lambda)
    scad_mm_pred <- sig(x_test %*% scad_mm_model_train_coef)
    scad_mm_pred_class <- as.numeric(scad_mm_pred > 0.5)
    
    # scad-lla
    scad_lla_model_coef <- scad_lla(x, y, scad_lambda)
    scad_lla_model_train_coef <- scad_lla(x_train, y_train, scad_lambda)
    scad_lla_pred <- sig(x_test %*% scad_lla_model_train_coef)
    scad_lla_pred_class <- as.numeric(scad_lla_pred > 0.5)
    
    # 自适应Lasso
    coef_lasso <- coef(lasso_model, s = cv_lasso_model$lambda.min)[-1]
    weights <- 1 / abs(coef_lasso)
    weights[is.infinite(weights)] <- 10^5   # 处理零系数
    cv_adaptive_lasso_model <- cv.glmnet(x, y, alpha = 1, penalty.factor = weights,family="binomial")
    adaptive_lasso_model <- glmnet(x, y, alpha = 1, penalty.factor = weights,family="binomial")
    coef_lasso_train <- coef(lasso_model_train, s = cv_lasso_model_train$lambda.min)[-1]
    weights_train <- 1 / abs(coef_lasso_train)
    weights_train[is.infinite(weights_train)] <- 10^5   # 处理零系数
    cv_adaptive_lasso_model_train <- cv.glmnet(x_train, y_train, alpha = 1, penalty.factor = weights_train,family="binomial")
    adaptive_lasso_model_train <- glmnet(x_train, y_train, alpha = 1, penalty.factor = weights_train,family="binomial")
    adaptive_lasso_pred <- predict(adaptive_lasso_model_train, s = cv_adaptive_lasso_model_train$lambda.min, newx = x_test, type = "response")
    adaptive_lasso_pred_class <- as.numeric(adaptive_lasso_pred > 0.5)
    
    # 弹性网
    cv_elastic_net_model <- cv.glmnet(x, y, family="binomial")
    elastic_net_model <- glmnet(x, y, family="binomial")
    cv_elastic_net_model_train <- cv.glmnet(x_train, y_train, family="binomial")
    elastic_net_model_train <- glmnet(x_train, y_train, family="binomial")
    elastic_net_pred <- predict( elastic_net_model_train, s = cv_elastic_net_model_train$lambda.min, newx = x_test, type = "response")
    elastic_net_pred_class <- as.numeric(elastic_net_pred > 0.5)
    
    # 计算指标
    # Lasso
    results$C_Lasso[sim] <- sum(coef(lasso_model, s = cv_lasso_model$lambda.min)[-1][1:5] != 0)
    results$IC_Lasso[sim] <- sum(coef(lasso_model, s = cv_lasso_model$lambda.min)[-1][6:p] != 0)
    results$EE_Lasso[sim] <- sqrt(sum((coef(lasso_model, s = cv_lasso_model$lambda.min)[-1] - beta_true[1:p])^2))
    results$errRate_Lasso[sim] <- mean(lasso_pred_class != y_test)
    lasso_pred_vector <- as.vector(lasso_pred)
    roc_result <- pROC::roc(y_test, lasso_pred_vector)  # 计算ROC曲线
    results$AUC_Lasso[sim] <- pROC::auc(roc_result)  # 计算AUC
    
    # Lasso-admm
    results$C_Lasso_admm[sim] <- sum(lasso_admm_model_coef[1:5] != 0)
    results$IC_Lasso_admm[sim] <- sum(lasso_admm_model_coef[6:p] != 0)
    results$EE_Lasso_admm[sim] <- sqrt(sum((lasso_admm_model_coef - beta_true[1:p])^2))
    results$errRate_Lasso_admm[sim] <- mean(lasso_admm_pred_class != y_test)
    lasso_admm_pred_vector <- as.vector(lasso_admm_pred)
    roc_result <- pROC::roc(y_test, lasso_admm_pred_vector)  # 计算ROC曲线
    results$AUC_Lasso_admm[sim] <- pROC::auc(roc_result)  # 计算AUC
    
    # Ridge
    results$C_Ridge[sim] <- sum(coef(ridge_model, s = cv_ridge_model$lambda.min)[-1][1:5] != 0)
    results$IC_Ridge[sim] <- sum(coef(ridge_model, s = cv_ridge_model$lambda.min)[-1][6:p] != 0)
    results$EE_Ridge[sim] <- sqrt(sum((coef(ridge_model, s = cv_ridge_model$lambda.min)[-1] - beta_true[1:p])^2))
    results$errRate_Ridge[sim] <- mean(ridge_pred_class != y_test)
    ridge_pred_vector <- as.vector(ridge_pred)
    roc_result <- pROC::roc(y_test, ridge_pred_vector)  # 计算ROC曲线
    results$AUC_Ridge[sim] <- pROC::auc(roc_result)  # 计算AUC
    
    # SCAD
    results$C_SCAD[sim] <- sum(coef(scad_model)[,cv_scad_model_lambdamin_index][-1][1:5] != 0)
    results$IC_SCAD[sim] <- sum(coef(scad_model)[,cv_scad_model_lambdamin_index][-1][6:p] != 0)
    results$EE_SCAD[sim] <- sqrt(sum((coef(scad_model)[,cv_scad_model_lambdamin_index][-1] - beta_true[1:p])^2))
    results$errRate_SCAD[sim] <- mean(scad_pred_class != y_test)
    scad_pred_vector <- as.vector(scad_pred)
    roc_result <- pROC::roc(y_test, scad_pred_vector)  # 计算ROC曲线
    results$AUC_SCAD[sim] <- pROC::auc(roc_result)  # 计算AUC
    
    # SCAD_lqa
    results$C_SCAD_lqa[sim] <- sum(scad_lqa_model_coef[1:5] != 0)
    results$IC_SCAD_lqa[sim] <- sum(scad_lqa_model_coef[6:p] != 0)
    results$EE_SCAD_lqa[sim] <- sqrt(sum((scad_lqa_model_coef - beta_true[1:p])^2))
    results$errRate_SCAD_lqa[sim] <- mean(scad_lqa_pred_class != y_test)
    scad_lqa_pred_vector <- as.vector(scad_lqa_pred)
    roc_result <- pROC::roc(y_test, scad_lqa_pred_vector)  # 计算ROC曲线
    results$AUC_SCAD_lqa[sim] <- pROC::auc(roc_result)  # 计算AUC
    
    # SCAD_mm
    results$C_SCAD_mm[sim] <- sum(scad_mm_model_coef[1:5] != 0)
    results$IC_SCAD_mm[sim] <- sum(scad_mm_model_coef[6:p] != 0)
    results$EE_SCAD_mm[sim] <- sqrt(sum((scad_mm_model_coef - beta_true[1:p])^2))
    results$errRate_SCAD_mm[sim] <- mean(scad_mm_pred_class != y_test)
    scad_mm_pred_vector <- as.vector(scad_mm_pred)
    roc_result <- pROC::roc(y_test, scad_mm_pred_vector)  # 计算ROC曲线
    results$AUC_SCAD_mm[sim] <- pROC::auc(roc_result)  # 计算AUC
    
    # SCAD_lla
    results$C_SCAD_lla[sim] <- sum(scad_lla_model_coef[1:5] != 0)
    results$IC_SCAD_lla[sim] <- sum(scad_lla_model_coef[6:p] != 0)
    results$EE_SCAD_lla[sim] <- sqrt(sum((scad_lla_model_coef - beta_true[1:p])^2))
    results$errRate_SCAD_lla[sim] <- mean(scad_lla_pred_class != y_test)
    scad_lla_pred_vector <- as.vector(scad_lla_pred)
    roc_result <- pROC::roc(y_test, scad_lla_pred_vector)  # 计算ROC曲线
    results$AUC_SCAD_lla[sim] <- pROC::auc(roc_result)  # 计算AUC
    
    
    # 自适应Lasso
    results$C_Adaptive_Lasso[sim] <- sum(coef(adaptive_lasso_model, s = cv_adaptive_lasso_model $lambda.min)[-1][1:5] != 0)
    results$IC_Adaptive_Lasso[sim] <- sum(coef(adaptive_lasso_model, s = cv_adaptive_lasso_model $lambda.min)[-1][6:p] != 0)
    results$EE_Adaptive_Lasso[sim] <- sqrt(sum((coef(adaptive_lasso_model, s = cv_adaptive_lasso_model$lambda.min)[-1] - beta_true[1:p])^2))
    results$errRate_Adaptive_Lasso[sim] <- mean(adaptive_lasso_pred_class != y_test)
    adaptive_lasso_pred_vector <- as.vector(adaptive_lasso_pred)
    roc_result <- pROC::roc(y_test, adaptive_lasso_pred_vector)  # 计算ROC曲线
    results$AUC_Adaptive_Lasso[sim] <- pROC::auc(roc_result)
    
    # 弹性网
    results$C_Elastic_Net[sim] <- sum(coef(elastic_net_model, s = cv_elastic_net_model$lambda.min)[-1][1:5] != 0)
    results$IC_Elastic_Net[sim] <- sum(coef(elastic_net_model, s = cv_elastic_net_model$lambda.min)[-1][6:p] != 0)
    results$EE_Elastic_Net[sim] <- sqrt(sum((coef(elastic_net_model, s = cv_elastic_net_model$lambda.min)[-1] - beta_true[1:p])^2))
    results$errRate_Elastic_Net[sim] <- mean(elastic_net_pred_class != y_test)
    elastic_net_pred_vector <- as.vector(elastic_net_pred)
    roc_result <- pROC::roc(y_test, elastic_net_pred_vector)  # 计算ROC曲线
    results$AUC_Elastic_Net[sim] <- pROC::auc(roc_result)
  }
  
  return(results)
}
