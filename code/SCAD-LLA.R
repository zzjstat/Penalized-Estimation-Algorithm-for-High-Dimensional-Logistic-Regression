# SCAD惩罚函数的导数
scad_derivative <- function(beta, lambda, a = 3.7) {
  if (abs(beta) <= lambda) {
    return(lambda)  # 当 |beta| <= lambda
  } else if (abs(beta) <= a * lambda) {
    return(a*lambda-beta / (a - 1))  # 当 lambda < |beta| <= a * lambda
  } else {
    return(0)  # 当 |beta| > a * lambda
  }
}
# X=x_train
# y=y_train
scad_lla<-function(X,y,lambda, max_iter = 500){
  n <- nrow(X)
  d <- ncol(X)
  # 初始估计
  cv_lasso_model <- cv.glmnet(X, y, alpha = 1,family="binomial") # 交叉验证确定最佳的 lambda 值
  lasso_model <- glmnet(X, y, alpha = 1,family="binomial")
  beta_lasso<-coef(lasso_model, s = cv_lasso_model$lambda.min)[-1]
  beta <- rep(0, d)
  beta_init <- tryCatch(
    { coef(glm(y~X,family = "binomial"))[-1]},
    warning = function(w){return(beta_lasso)},
    error = function(e) { return(beta_lasso) }
  )  # OLS初始估计
  beta <- as.vector(beta_init)
  for (iter in 1:max_iter) {
    beta_old <- beta
    weights <- sapply(beta, function(b) scad_derivative(abs(b), lambda))
    # 使用 tryCatch 处理可能的错误
    cv_adaptive_lasso_model <- tryCatch(
      cv.glmnet(X, y, alpha = 1, penalty.factor = weights, family = "binomial"),
      error = function(e) {
        message("Error in cv.glmnet: ", e$message)  # 输出错误信息
        return(NULL)  # 返回 NULL，表示出现错误
      }
    )
    # 检查是否成功运行 cv.glmnet
    if (is.null(cv_adaptive_lasso_model)) {
      break  # 如果 cv_adaptive_lasso_model 为 NULL，则退出循环
    }
    
    adaptive_lasso_model <- glmnet(X, y, alpha = 1, penalty.factor = weights,family="binomial")
    # 获取当前的 beta 系数
    beta<-coef(adaptive_lasso_model, s = cv_adaptive_lasso_model $lambda.min)[-1]
    # 步骤 4: 收敛检查
    if (sum(abs(beta - beta_old)) < 1e-4) {
      break
    }
  }
  return(beta)
}
