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

# 逻辑回归的负对数似然损失函数
logistic_nll <- function(beta, X, y) {
  linear_pred <- X %*% beta  # 线性预测
  prob <- 1 / (1 + exp(-linear_pred))  # 计算sigmoid概率
  nll <- -sum(y * log(prob + 1e-10) + (1 - y) * log(1 - prob + 1e-10))
  return(1/nrow(X)*nll)
}

# 计算梯度
logistic_gradient <- function(beta, X, y) {
  p <- 1 / (1 + exp(-X %*% beta))
  return(t(X) %*% (p-y))  # 返回梯度
}

# Hessian矩阵
logistic_hessian <- function(beta, X) {
  p <- 1 / (1 + exp(-X %*% beta))
  W <- diag(as.vector(p * (1 - p)))  # 权重矩阵
  return(t(X) %*% W %*% X)  # 返回 Hessian
}


# MM算法
scad_mm <- function(X, y, lambda, max_iter = 500) {
  eta = 1e-2
  tau = 1e-2
  n <- nrow(X)
  d <- ncol(X)
  epsilon <- 1e-6  # 正则化参数
  # 步骤 1: 初始估计
  cv_lasso_model <- cv.glmnet(X, y, family="binomial") # 交叉验证确定最佳的 lambda 值
  lasso_model <- glmnet(X, y,family="binomial")
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
    # 步骤 2: 局部二次近似
    grad <- logistic_gradient(beta, X, y) + n * (sapply(beta, function(b) b*scad_derivative(abs(b), lambda)/(abs(b)+tau)))
    hess <- logistic_hessian(beta, X) + epsilon * diag(ncol(X)) + n * diag(sapply(beta, function(b) scad_derivative(abs(b), lambda) / (abs(b)+tau)))
    # 计算更新
    beta <- beta - solve(hess) %*% grad
    beta[which(abs(beta)<eta)] <- 0
    # 步骤 4: 收敛检查
    if (sum(abs(beta - beta_old)) < 1e-4) {
      break
    }
  }
  return(beta)
}

